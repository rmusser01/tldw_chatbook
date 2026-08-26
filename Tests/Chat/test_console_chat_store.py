import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyDefaults,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_roleplay_identity import (
    resolve_console_message_presentation,
)
from tldw_chatbook.Chat.console_roleplay_metadata import ConsoleRoleplayContext
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem, read_conversation_scope
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Sync_Interop.chat_outbox_producer import ChatSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope
from tldw_chatbook.TTS.profile_types import CharacterRef
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID, LocalWorkspaceRegistryService


def _pristine_defaults(*, model: str = "default-model") -> ConsoleSessionSettings:
    return ConsoleSessionSettings(provider="openai", model=model)


def _library_authority(
    attempt_id: str,
    *,
    auto_retrieve: ConsoleAutoRetrieve = ConsoleAutoRetrieve.AUTOMATIC,
    assistant_access: ConsoleAssistantLibraryAccess = (
        ConsoleAssistantLibraryAccess.BLOCKED
    ),
) -> ConsoleTurnLibraryAuthority:
    return ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=auto_retrieve,
            assistant_access=assistant_access,
            policy_revision=1,
            source="durable",
        ),
        direct_library_tools=True,
        source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
        scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), True),
        provider_intent=ConsoleProviderIntent("openai", "model-a", None),
        attempt_id=attempt_id,
    )


def _begin_disclosed_library_attempt(
    store: ConsoleChatStore,
    session_id: str,
    *,
    attempt_id: str = "attempt-active",
    content: str = "",
) -> tuple[ConsoleChatMessage, ConsoleResolvedDestination]:
    local = ConsoleResolvedDestination(
        provider="llama_cpp",
        model="model-a",
        endpoint_identity="http://127.0.0.1:9099",
        egress_class=ConsoleEgressClass.ON_DEVICE,
    )
    external = ConsoleResolvedDestination(
        provider="openai",
        model="model-a",
        endpoint_identity="https://api.openai.com",
        egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
    )
    baseline = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.begin_session_library_destination_attempt(
        session_id,
        _library_authority("attempt-baseline"),
        local,
        baseline.id,
    )
    store.append_stream_chunk(baseline.id, "baseline")
    store.mark_message_complete(baseline.id)
    assistant = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
    )
    store.begin_session_library_destination_attempt(
        session_id,
        _library_authority(attempt_id),
        external,
        assistant.id,
    )
    return assistant, external


def _pristine_session(
    store: ConsoleChatStore,
    defaults: ConsoleSessionSettings,
    **kwargs,
) -> ConsoleChatSession:
    return store.ensure_session(
        title="Chat 1",
        settings=defaults,
        canonical_settings_baseline=defaults,
        **kwargs,
    )


def test_create_session_rejects_mismatched_canonical_provenance():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()

    with pytest.raises(ValueError, match="canonical baseline"):
        store.create_session(
            settings=defaults,
            canonical_settings_baseline=replace(defaults, model="not-the-snapshot"),
        )

    assert store.sessions() == []


def test_initial_chat_one_is_pristine_until_the_user_types():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)

    assert store.is_pristine_session(session.id, expected_settings=defaults)

    store.set_session_draft(session.id, "typed work")
    assert not store.is_pristine_session(session.id, expected_settings=defaults)


def test_default_library_policy_does_not_dirty_pristine_tab_but_explicit_edit_does():
    defaults = _pristine_defaults()
    store = ConsoleChatStore(
        library_policy_defaults=ConsoleLibraryPolicyDefaults(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        )
    )
    session = _pristine_session(store, defaults)

    assert store.is_pristine_session(session.id, expected_settings=defaults)

    store.stage_session_library_policy(
        session.id,
        ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )

    assert not store.is_pristine_session(session.id, expected_settings=defaults)


def test_message_completed_subscription_emits_first_live_completion_once():
    store = ConsoleChatStore()
    session = store.create_session()
    observed: list[tuple[str, str]] = []
    unsubscribe = store.subscribe_message_completed(observed.append)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )

    store.append_stream_chunk(message.id, "Welcome back.")
    completed = store.mark_message_complete(message.id)

    assert observed == [(session.id, completed.id)]
    assert type(observed[0]) is tuple
    unsubscribe()


def test_message_completed_subscription_ignores_complete_append_and_unsubscribe():
    store = ConsoleChatStore()
    session = store.create_session()
    observed: list[tuple[str, str]] = []
    unsubscribe = store.subscribe_message_completed(observed.append)

    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Existing greeting.",
    )
    unsubscribe()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.append_stream_chunk(message.id, "New reply.")
    store.mark_message_complete(message.id)

    assert observed == []


def test_message_completed_subscription_isolates_callback_failure_and_duplicate_terminalization():
    store = ConsoleChatStore()
    session = store.create_session()
    observed: list[tuple[str, str]] = []

    def raising_callback(_token: tuple[str, str]) -> None:
        raise RuntimeError("subscriber failed")

    store.subscribe_message_completed(raising_callback)
    store.subscribe_message_completed(observed.append)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.append_stream_chunk(message.id, "New reply.")

    store.mark_message_complete(message.id)
    with pytest.raises(ValueError):
        store.mark_message_complete(message.id)

    assert observed == [(session.id, message.id)]


def test_message_completed_subscription_emits_each_successful_regeneration() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    observed: list[tuple[str, str]] = []
    store.subscribe_message_completed(observed.append)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Original.",
    )

    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "First regeneration.")
    store.finalize_variant_stream(message.id)
    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "Second regeneration.")
    store.finalize_variant_stream(message.id)

    assert observed == [
        (session.id, message.id),
        (session.id, message.id),
    ]


@pytest.mark.parametrize(
    "terminal",
    ["complete", "failed", "stopped", "variant_complete"],
)
def test_assistant_terminal_settlement_clears_runtime_library_disclosure(
    terminal: str,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message, external = _begin_disclosed_library_attempt(
        store,
        session.id,
        content="original" if terminal == "variant_complete" else "",
    )
    assert session.library_destination_runtime.disclosure is not None
    assert session.library_destination_runtime.owner_attempt_id == "attempt-active"
    assert session.library_destination_runtime.owner_message_id == message.id

    if terminal == "variant_complete":
        store.begin_variant_stream(message.id)
        store.append_stream_chunk(message.id, "replacement")
        store.finalize_variant_stream(message.id)
    else:
        store.append_stream_chunk(message.id, "response")
        getattr(store, f"mark_message_{terminal}")(message.id)

    assert session.library_destination_runtime.disclosure is None
    assert session.library_destination_runtime.owner_attempt_id is None
    assert session.library_destination_runtime.owner_message_id is None
    assert session.library_destination_runtime.resolved_destination == external
    assert session.library_destination_runtime.last_resolved_identity == (
        external.identity_key
    )


def test_completion_subscribers_observe_disclosure_already_settled() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message, _external = _begin_disclosed_library_attempt(store, session.id)
    observed = []
    store.subscribe_message_completed(
        lambda _token: observed.append(session.library_destination_runtime.disclosure)
    )
    store.append_stream_chunk(message.id, "response")

    store.mark_message_complete(message.id)

    assert observed == [None]


def test_older_completed_variant_cannot_settle_a_newer_attempt_disclosure() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    older = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="older answer",
    )
    active, _external = _begin_disclosed_library_attempt(store, session.id)
    disclosure = session.library_destination_runtime.disclosure
    assert disclosure is not None

    store.add_variant(older.id, "older alternate")

    assert session.library_destination_runtime.disclosure == disclosure
    assert session.library_destination_runtime.owner_attempt_id == "attempt-active"
    assert session.library_destination_runtime.owner_message_id == active.id


def test_library_destination_settlement_requires_exact_attempt_and_message_owner() -> (
    None
):
    store = ConsoleChatStore()
    session = store.create_session()
    active, _external = _begin_disclosed_library_attempt(store, session.id)
    disclosure = session.library_destination_runtime.disclosure

    wrong_attempt = store.settle_session_library_destination(
        session.id,
        expected_attempt_id="attempt-older",
        expected_message_id=active.id,
    )
    wrong_message = store.settle_session_library_destination(
        session.id,
        expected_attempt_id="attempt-active",
        expected_message_id="older-message",
    )

    assert wrong_attempt.disclosure == disclosure
    assert wrong_message.disclosure == disclosure
    settled = store.settle_session_library_destination(
        session.id,
        expected_attempt_id="attempt-active",
        expected_message_id=active.id,
    )
    assert settled.disclosure is None
    assert settled.owner_attempt_id is None
    assert settled.owner_message_id is None


def test_older_attempt_cleanup_cannot_clear_replacement_destination_owner() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    older, _external = _begin_disclosed_library_attempt(store, session.id)
    replacement = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    private = ConsoleResolvedDestination(
        provider="custom",
        model="model-b",
        endpoint_identity="http://10.0.0.9:8080",
        egress_class=ConsoleEgressClass.PRIVATE_NETWORK,
    )
    store.begin_session_library_destination_attempt(
        session.id,
        _library_authority("attempt-replacement"),
        private,
        replacement.id,
    )

    store.settle_session_library_destination(
        session.id,
        expected_attempt_id="attempt-active",
        expected_message_id=older.id,
    )

    runtime = session.library_destination_runtime
    assert runtime.disclosure is not None
    assert runtime.disclosure.resolved_destination == private
    assert runtime.owner_attempt_id == "attempt-replacement"
    assert runtime.owner_message_id == replacement.id


def test_runtime_library_disclosure_is_isolated_across_session_navigation() -> None:
    store = ConsoleChatStore()
    session_a = store.create_session(title="Session A")
    session_b = store.create_session(title="Session B")
    message_a, _external_a = _begin_disclosed_library_attempt(
        store,
        session_a.id,
        attempt_id="attempt-a",
    )
    message_b, _external_b = _begin_disclosed_library_attempt(
        store,
        session_b.id,
        attempt_id="attempt-b",
    )

    store.switch_session(session_b.id)
    store.switch_session(session_a.id)
    assert session_a.library_destination_runtime.disclosure is not None
    assert session_b.library_destination_runtime.disclosure is not None

    store.append_stream_chunk(message_b.id, "response")
    store.mark_message_complete(message_b.id)

    assert store.active_session_id == session_a.id
    assert session_a.library_destination_runtime.disclosure is not None
    assert session_a.library_destination_runtime.owner_message_id == message_a.id
    assert session_b.library_destination_runtime.disclosure is None


def test_completion_generation_remains_monotonic_across_same_id_restore() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Original.",
    )
    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "Before restore.")
    store.finalize_variant_stream(message.id)
    before_restore = store.message_completion_generation(message.id)
    restored_messages = store.messages_for_session(session.id)

    store.restore_state(
        sessions=[replace(session)],
        messages_by_session={session.id: restored_messages},
        active_session_id=session.id,
    )
    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "After restore.")
    store.finalize_variant_stream(message.id)

    assert store.message_completion_generation(message.id) > before_restore


def test_message_completed_subscription_add_variant_emits_but_selection_does_not() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    observed: list[tuple[str, str]] = []
    store.subscribe_message_completed(observed.append)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Original.",
    )

    store.add_variant(message.id, "Regenerated.")
    store.select_variant(message.id, 0)

    assert observed == [(session.id, message.id)]


def test_message_completed_subscription_duplicate_variant_finalize_fails_closed() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    observed: list[tuple[str, str]] = []
    store.subscribe_message_completed(observed.append)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Original.",
    )
    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "Regenerated.")
    store.finalize_variant_stream(message.id)

    with pytest.raises(ValueError, match="active variant stream"):
        store.finalize_variant_stream(message.id)

    assert observed == [(session.id, message.id)]


def test_reply_speech_preference_disqualifies_initial_session_reuse():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)

    store.set_auto_speak(session.id, True)

    assert not store.is_pristine_session(session.id, expected_settings=defaults)


def test_typed_then_cleared_session_keeps_durable_work_marker():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)

    store.set_session_draft(session.id, "typed work")
    store.set_session_draft(session.id, "")

    assert not store.is_pristine_session(session.id, expected_settings=defaults)


def test_pristine_session_rejects_orphan_message_ownership_index():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)
    message_id = "orphan-message-owned-by-pristine-session"
    store._message_session_index[message_id] = session.id

    assert not store.is_pristine_session(session.id, expected_settings=defaults)


def test_pristine_session_rejects_owned_tree_node_outside_visible_message_list():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)
    hidden_message = ConsoleChatMessage(
        id="hidden-owned-message",
        role=ConsoleMessageRole.ASSISTANT,
        content="hidden work",
    )
    store._register_tree_node(session.id, hidden_message, parent_native_id=None)
    assert store.messages_for_session(session.id) == []

    assert not store.is_pristine_session(session.id, expected_settings=defaults)


def test_pristine_session_does_not_assign_unattributed_message_cache_state():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)
    store._stream_chunks_by_message["unattributed-message"] = ["foreign chunk"]

    assert store.is_pristine_session(session.id, expected_settings=defaults)


def test_pristine_session_allows_harmless_initialized_empty_cache_entries():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)
    store._tool_markers_by_session[session.id] = []
    store._roleplay_system_projection_candidates[session.id] = ()
    store._payload_revisions[session.id] = 1

    assert store.is_pristine_session(session.id, expected_settings=defaults)


@pytest.mark.parametrize(
    "disqualify",
    [
        pytest.param(
            lambda store, session: setattr(session, "title", "Chat 2"), id="title"
        ),
        pytest.param(
            lambda store, session: setattr(
                session, "persisted_conversation_id", "conversation-1"
            ),
            id="persisted-conversation",
        ),
        pytest.param(
            lambda store, session: store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="hello",
                persist=False,
            ),
            id="message",
        ),
        pytest.param(
            lambda store, session: store._nodes_by_session[session.id].update(
                {"orphan": object()}
            ),
            id="off-path-tree-node",
        ),
        pytest.param(
            lambda store, session: setattr(session, "draft", "draft"), id="draft"
        ),
        pytest.param(
            lambda store, session: session.pending_attachments.append(object()),
            id="attachment",
        ),
        pytest.param(
            lambda store, session: setattr(session, "one_shot_prefill", "prefill"),
            id="one-shot-prefill",
        ),
        pytest.param(
            lambda store, session: setattr(
                session,
                "settings",
                replace(session.settings, pinned_prefill="Always:"),
            ),
            id="pinned-prefill",
        ),
        pytest.param(
            lambda store, session: session.rag_scope_holder.set(
                RagScope(
                    items=(ScopeItem("media", "m1"),),
                    updated_at="2026-01-01T00:00:00Z",
                )
            ),
            id="rag-scope",
        ),
        pytest.param(
            lambda store, session: setattr(
                session,
                "context_policy_overrides",
                ConsoleContextPolicyOverrides(summary_max_tokens=256),
            ),
            id="context-overrides",
        ),
        pytest.param(
            lambda store, session: setattr(session, "context_policy_error", "bad"),
            id="context-error",
        ),
        pytest.param(
            lambda store, session: setattr(session, "runtime_backend", "server"),
            id="runtime-backend",
        ),
        pytest.param(
            lambda store, session: setattr(session, "assistant_kind", "character"),
            id="assistant-kind",
        ),
        pytest.param(
            lambda store, session: setattr(session, "assistant_id", "7"),
            id="assistant-id",
        ),
        pytest.param(
            lambda store, session: setattr(
                session, "assistant_authority_id", "authority"
            ),
            id="assistant-authority",
        ),
        pytest.param(
            lambda store, session: setattr(session, "character_id", 7),
            id="character-id",
        ),
        pytest.param(
            lambda store, session: setattr(session, "character_name", "Alba"),
            id="character-name",
        ),
        pytest.param(
            lambda store, session: setattr(
                session, "user_display_name_override", "Captain"
            ),
            id="user-name-override",
        ),
        pytest.param(
            lambda store, session: setattr(
                session, "character_system_template", "Stay in character."
            ),
            id="character-template",
        ),
        pytest.param(
            lambda store, session: setattr(session, "identity_revision", 1),
            id="identity-revision",
        ),
        pytest.param(
            lambda store, session: setattr(session, "ephemeral", True),
            id="ephemeral",
        ),
        pytest.param(
            lambda store, session: setattr(
                session, "settings", _pristine_defaults(model="changed-model")
            ),
            id="altered-settings",
        ),
        pytest.param(
            lambda store, session: setattr(session, "settings", None),
            id="missing-settings",
        ),
        pytest.param(
            lambda store, session: session.todo_store.create(content="work"),
            id="todo-work",
        ),
        pytest.param(
            lambda store, session: store._tool_markers_by_session.update(
                {session.id: [(None, object())]}
            ),
            id="tool-state",
        ),
        pytest.param(
            lambda store, session: store._context_summary_by_session.update(
                {session.id: ("summary", None)}
            ),
            id="context-summary",
        ),
        pytest.param(
            lambda store, session: store._roleplay_system_projection_candidates.update(
                {session.id: ("prompt",)}
            ),
            id="roleplay-work-state",
        ),
        pytest.param(
            lambda store, session: store._conversation_context_epochs.update(
                {session.id: 1}
            ),
            id="provider-context-work-state",
        ),
    ],
)
def test_pristine_session_rejects_each_work_or_identity_disqualifier(disqualify):
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)

    disqualify(store, session)

    assert not store.is_pristine_session(session.id, expected_settings=defaults)


def test_pristine_session_predicate_returns_false_for_missing_session():
    store = ConsoleChatStore()

    assert not store.is_pristine_session(
        "missing", expected_settings=_pristine_defaults()
    )


def test_refresh_pristine_session_settings_preserves_live_object_identity():
    prior = _pristine_defaults(model="stale-model")
    current = _pristine_defaults(model="current-model")
    store = ConsoleChatStore()
    session = _pristine_session(store, prior)
    updated_at_before = session.updated_at

    refreshed = store.refresh_pristine_session_settings(
        session.id,
        prior_canonical_settings=prior,
        current_canonical_settings=current,
    )

    assert refreshed is session
    assert store.sessions()[0] is session
    assert session.settings is current
    assert session.canonical_settings_baseline is current
    assert session.updated_at != updated_at_before


@pytest.mark.parametrize(
    "prior_change,current_change",
    [
        pytest.param({"source": "user"}, {}, id="prior-not-derived"),
        pytest.param({}, {"source": "user"}, id="current-not-derived"),
        pytest.param({}, {"system_prompt": "custom"}, id="current-system-prompt"),
        pytest.param({}, {"character_label": "Alba"}, id="current-character-label"),
        pytest.param({}, {"pinned_prefill": "Always:"}, id="current-prefill"),
    ],
)
def test_refresh_pristine_session_settings_rejects_nondefault_baselines(
    prior_change,
    current_change,
):
    canonical_prior = _pristine_defaults(model="stale-model")
    prior = replace(canonical_prior, **prior_change)
    current = replace(
        _pristine_defaults(model="current-model"),
        **current_change,
    )
    store = ConsoleChatStore()
    session = _pristine_session(store, prior)
    before = replace(session)

    with pytest.raises(ValueError, match="derived defaults"):
        store.refresh_pristine_session_settings(
            session.id,
            prior_canonical_settings=prior,
            current_canonical_settings=current,
        )

    assert store.sessions()[0] is session
    assert session == before


def test_refresh_pristine_session_settings_revalidation_failure_is_nonmutating(
    monkeypatch,
):
    prior = _pristine_defaults(model="stale-model")
    current = _pristine_defaults(model="current-model")
    store = ConsoleChatStore()
    session = _pristine_session(store, prior)
    before = replace(session)
    monkeypatch.setattr(store, "is_pristine_session", lambda *_args, **_kwargs: False)

    with pytest.raises(ValueError, match="pristine"):
        store.refresh_pristine_session_settings(
            session.id,
            prior_canonical_settings=prior,
            current_canonical_settings=current,
        )

    assert store.sessions()[0] is session
    assert session == before


def test_repurpose_pristine_session_preserves_slot_and_applies_identity_atomically():
    defaults = _pristine_defaults()
    roleplay_settings = replace(
        defaults,
        system_prompt="You are Alba.",
        character_label="Alba",
    )
    store = ConsoleChatStore()
    first = store.create_session(
        title="Other", workspace_id="workspace-before", settings=defaults
    )
    target = store.create_session(
        title="Chat 1",
        workspace_id="workspace-target",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    order_before = [session.id for session in store.sessions()]
    updated_at_before = target.updated_at
    payload_revision_before = store.payload_revision(target.id)
    identity_revision_before = target.identity_revision

    updated = store.repurpose_pristine_session(
        target.id,
        canonical_settings=defaults,
        trusted_system_prompt="You are Alba.",
        title="Chat with Alba",
        settings=roleplay_settings,
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
        character_name="Alba",
    )

    assert updated is target
    assert store.sessions()[1] is target
    assert updated.id == target.id
    assert updated.workspace_id == "workspace-target"
    assert [session.id for session in store.sessions()] == order_before
    assert store.sessions()[0] is first
    assert updated.title == "Chat with Alba"
    assert updated.settings == roleplay_settings
    assert updated.runtime_backend == "local"
    assert updated.assistant_kind == "character"
    assert updated.assistant_id == "7"
    assert updated.assistant_authority_id == "local-authority"
    assert updated.character_id == 7
    assert updated.character_name == "Alba"
    assert updated.persisted_conversation_id is None
    assert updated.canonical_settings_baseline is None
    assert updated.updated_at != updated_at_before
    assert updated.identity_revision == identity_revision_before + 1
    assert store.payload_revision(updated.id) == payload_revision_before + 1
    presentation = store.presentation_context(updated.id, "User")
    assert presentation.character_name == "Alba"
    assert presentation.assistant_kind == "character"
    assert presentation.revision == updated.identity_revision


@pytest.mark.parametrize(
    "system_template,greeting_template,expected_identity_revision,expected_payload_revision",
    [
        pytest.param("", "", 1, 1, id="identity-only"),
        pytest.param("Stay {{char}}.", "", 2, 2, id="template"),
        pytest.param("Stay {{char}}.", "Hello.", 2, 3, id="template-and-greeting"),
    ],
)
def test_repurpose_and_seed_revision_contracts(
    system_template,
    greeting_template,
    expected_identity_revision,
    expected_payload_revision,
):
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)

    store.repurpose_pristine_session(
        session.id,
        canonical_settings=defaults,
        trusted_system_prompt="You are Alba.",
        title="Chat with Alba",
        settings=replace(
            defaults,
            system_prompt="You are Alba.",
            character_label="Alba",
        ),
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
        character_name="Alba",
    )
    store.seed_character_roleplay(
        session.id,
        system_template=system_template,
        greeting_template=greeting_template,
        global_default="User",
    )

    assert session.identity_revision == expected_identity_revision
    assert store.payload_revision(session.id) == expected_payload_revision
    assert store.presentation_context(session.id, "User").revision == (
        expected_identity_revision
    )
    assert len(store.messages_for_session(session.id)) == int(bool(greeting_template))


def test_repurpose_pristine_session_revalidation_failure_is_nonmutating(monkeypatch):
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)
    before = replace(session)
    payload_revision_before = store.payload_revision(session.id)
    payload_revisions_before = dict(store._payload_revisions)
    monkeypatch.setattr(store, "is_pristine_session", lambda *_args, **_kwargs: False)

    with pytest.raises(ValueError, match="pristine"):
        store.repurpose_pristine_session(
            session.id,
            canonical_settings=defaults,
            trusted_system_prompt="You are Alba.",
            title="Chat with Alba",
            settings=replace(
                defaults,
                system_prompt="You are Alba.",
                character_label="Alba",
            ),
            runtime_backend="local",
            assistant_kind="character",
            assistant_id="7",
            assistant_authority_id="local-authority",
            character_id=7,
            character_name="Alba",
        )

    assert store.sessions() == [before]
    assert store.sessions()[0] is session
    assert session == before
    assert store.payload_revision(session.id) == payload_revision_before
    assert store._payload_revisions == payload_revisions_before


@pytest.mark.parametrize(
    "settings_change",
    [
        pytest.param({"provider": "anthropic"}, id="provider"),
        pytest.param({"model": "different-model"}, id="model"),
        pytest.param({"source": "user"}, id="source"),
        pytest.param({"temperature": 0.1}, id="temperature"),
        pytest.param({"pinned_prefill": "Always:"}, id="pinned-prefill"),
        pytest.param({"character_label": "Not Alba"}, id="character-label"),
        pytest.param({"system_prompt": "Arbitrary prompt"}, id="system-prompt"),
    ],
)
def test_repurpose_rejects_noncanonical_roleplay_settings_without_mutation(
    settings_change,
):
    defaults = _pristine_defaults()
    trusted_prompt = "You are Alba."
    valid_settings = replace(
        defaults,
        system_prompt=trusted_prompt,
        character_label="Alba",
    )
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)
    before = replace(session)
    payload_revision_before = store.payload_revision(session.id)
    payload_revisions_before = dict(store._payload_revisions)

    with pytest.raises(ValueError):
        store.repurpose_pristine_session(
            session.id,
            canonical_settings=defaults,
            trusted_system_prompt=trusted_prompt,
            title="Chat with Alba",
            settings=replace(valid_settings, **settings_change),
            runtime_backend="local",
            assistant_kind="character",
            assistant_id="7",
            assistant_authority_id="local-authority",
            character_id=7,
            character_name="Alba",
        )

    assert store.sessions() == [before]
    assert store.sessions()[0] is session
    assert session == before
    assert store.payload_revision(session.id) == payload_revision_before
    assert store._payload_revisions == payload_revisions_before


def test_repurpose_rejects_mismatched_roleplay_title_without_mutation():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)
    before = replace(session)

    with pytest.raises(ValueError):
        store.repurpose_pristine_session(
            session.id,
            canonical_settings=defaults,
            trusted_system_prompt="You are Alba.",
            title="Alba roleplay",
            settings=replace(
                defaults,
                system_prompt="You are Alba.",
                character_label="Alba",
            ),
            runtime_backend="local",
            assistant_kind="character",
            assistant_id="7",
            assistant_authority_id="local-authority",
            character_id=7,
            character_name="Alba",
        )

    assert store.sessions() == [before]


def test_session_character_ref_projects_complete_local_and_server_identities():
    local = ConsoleChatSession(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
    )
    server = ConsoleChatSession(
        runtime_backend="server",
        assistant_kind="character",
        assistant_id="opaque-character",
        assistant_authority_id="server-user-v1:" + ("a" * 64),
        character_id=None,
    )

    assert local.character_ref() == CharacterRef(
        source="local",
        authority_id="local-authority",
        character_id="7",
    )
    assert server.character_ref() == CharacterRef(
        source="server",
        authority_id="server-user-v1:" + ("a" * 64),
        character_id="opaque-character",
    )


def test_session_local_character_id_requires_canonical_local_character_identity():
    valid = ConsoleChatSession(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
    )
    invalid = (
        ConsoleChatSession(
            runtime_backend="server",
            assistant_kind="character",
            assistant_id="opaque-character",
            character_id=7,
        ),
        ConsoleChatSession(
            runtime_backend="local",
            assistant_kind="persona",
            assistant_id="7",
            character_id=7,
        ),
        ConsoleChatSession(
            runtime_backend="local",
            assistant_kind="character",
            assistant_id="1",
            character_id=True,
        ),
        ConsoleChatSession(
            runtime_backend="local",
            assistant_kind="character",
            assistant_id="0",
            character_id=0,
        ),
        ConsoleChatSession(
            runtime_backend="local",
            assistant_kind="character",
            assistant_id="007",
            character_id=7,
        ),
    )

    assert valid.local_character_id() == 7
    assert all(session.local_character_id() is None for session in invalid)


@pytest.mark.parametrize(
    "session_kwargs",
    [
        {
            "runtime_backend": "server",
            "assistant_kind": "character",
            "assistant_id": "opaque-character",
            "assistant_authority_id": None,
        },
        {
            "runtime_backend": "local",
            "assistant_kind": "persona",
            "assistant_id": "persona-1",
            "assistant_authority_id": None,
        },
        {},
        {
            "runtime_backend": "local",
            "assistant_kind": "character",
            "assistant_id": "007",
            "assistant_authority_id": "local-authority",
            "character_id": 7,
        },
        {
            "runtime_backend": "server",
            "assistant_kind": "character",
            "assistant_id": "opaque-character",
            "assistant_authority_id": "server-authority",
            "character_id": 7,
        },
        {
            "runtime_backend": "other",
            "assistant_kind": "character",
            "assistant_id": "7",
            "assistant_authority_id": "local-authority",
            "character_id": 7,
        },
    ],
    ids=[
        "unscoped-server",
        "persona",
        "generic",
        "mismatched-local-id",
        "server-with-local-projection",
        "unknown-source",
    ],
)
def test_session_character_ref_rejects_unproven_or_inconsistent_identity(
    session_kwargs,
):
    session = ConsoleChatSession(**session_kwargs)
    assert session.character_ref() is None


def test_store_creates_session_and_appends_messages():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1", workspace_id="global")

    user_message = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hello"
    )
    assistant_message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    assert store.active_session_id == session.id
    assert user_message.content == "hello"
    assert assistant_message.status == "pending"
    assert [message.role for message in store.messages_for_session(session.id)] == [
        ConsoleMessageRole.USER,
        ConsoleMessageRole.ASSISTANT,
    ]


def test_session_is_ephemeral_mirrors_session_workspace_id():
    """F4 (final-review): the accessor `ConsoleAgentBridge.run_reply` uses
    to thread a session's temporary flag into `BuiltinToolProvider`, mirrors
    `session_workspace_id`'s own shape exactly (raises `KeyError` for an
    unknown session id -- callers degrade that to `False`, never let it
    escape)."""
    store = ConsoleChatStore()
    normal = store.create_session(title="Normal")
    temp = store.create_session(title="Temp", ephemeral=True)

    assert store.session_is_ephemeral(normal.id) is False
    assert store.session_is_ephemeral(temp.id) is True
    with pytest.raises(KeyError):
        store.session_is_ephemeral("no-such-session")


def test_store_records_message_feedback():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )

    updated = store.set_message_feedback(message.id, "up")

    assert updated.feedback == "up"
    assert store.get_message(message.id).feedback == "up"


def test_store_deletes_message_from_transcript():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )

    deleted = store.delete_message(message.id)

    assert deleted.id == message.id
    assert store.messages_for_session(session.id) == []
    with pytest.raises(KeyError):
        store.get_message(message.id)


def test_stop_mid_regenerate_restores_base_and_does_not_orphan_it():
    """Plan-B final-review Medium-2: stopping a message mid variant-stream
    (regenerate) must restore the pre-regenerate base content AND status --
    mirroring ``mark_message_failed`` (Plan-B Task 1) -- and pop the base
    immediately, rather than leaving it orphaned in `_variant_stream_bases`
    for `delete_message` to clean up later. (This test previously pinned
    the opposite, buggy behavior: that a stopped regenerate replaced the
    original answer with the partial stream and left the base to be
    cleared only by a later delete -- Plan-B Task 1 Minor finding's fix
    only covered `delete_message` itself, not this root cause.)"""
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "partial")

    stopped = store.mark_message_stopped(message.id)

    assert stopped.content == "answer"
    assert stopped.status == "complete"
    assert message.id not in store._variant_stream_bases

    # The now-terminal message can still be deleted cleanly afterward.
    store.delete_message(message.id)
    with pytest.raises(KeyError):
        store.get_message(message.id)


def test_stop_mid_regenerate_leaves_existing_variants_untouched():
    """Plan-B final-review Medium-2: a stopped regenerate must not disturb
    variants recorded by earlier, successfully-finalized regenerates."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="v1"
    )
    store.add_variant(message.id, "v2")
    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "v3-partial")

    stopped = store.mark_message_stopped(message.id)

    assert stopped.content == "v2"
    assert stopped.status == "complete"
    assert [v.content for v in stopped.variants.variants] == ["v1", "v2"]
    assert stopped.variants.selected_index == 1
    assert message.id not in store._variant_stream_bases


def test_stop_mid_plain_send_keeps_partial_content_and_stopped_status():
    """Plan-B final-review Medium-2: a Stop with no captured variant base
    (a normal, non-regenerate send) must keep today's behavior unchanged --
    the partial streamed content is kept and the message is marked
    "stopped", not silently reverted."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(message.id, "par")
    store.append_stream_chunk(message.id, "tial")

    stopped = store.mark_message_stopped(message.id)

    assert stopped.content == "partial"
    assert stopped.status == "stopped"
    assert message.id not in store._variant_stream_bases


def test_store_updates_message_content():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )

    updated = store.update_message_content(message.id, "edited answer")

    assert updated.content == "edited answer"
    assert store.get_message(message.id).content == "edited answer"


def test_store_updates_current_variant_content():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="first"
    )
    store.add_variant(message.id, "second")

    updated = store.update_message_content(message.id, "edited second")

    assert updated.content == "edited second"
    assert updated.variants is not None
    assert updated.variants.selected_index == 1
    assert updated.variants.current.content == "edited second"
    assert updated.variants.variants[0].content == "first"


def test_store_updates_streaming_message_and_marks_stopped():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    store.append_stream_chunk(message.id, "hel")
    store.append_stream_chunk(message.id, "lo")
    store.mark_message_stopped(message.id)

    updated = store.get_message(message.id)
    assert updated.content == "hello"
    assert updated.status == "stopped"


def test_store_buffers_stream_chunks_until_messages_are_materialized():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    chunk_result = store.append_stream_chunk(message.id, "hel")

    assert chunk_result.content == ""
    materialized = store.messages_for_session(session.id)[0]
    assert materialized.content == "hel"
    assert materialized.status == "streaming"


def test_reset_stream_content_discards_leaked_prose_but_keeps_streaming_status():
    """Plan-B Task 5 Finding A: once a streamed turn is classified as a tool
    call, any prose already streamed to the store for it must be discarded
    so the next turn's chunks start clean instead of concatenating onto
    already-flushed leaked prose."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    store.append_stream_chunk(message.id, "Let me check that for you.")
    reset = store.reset_stream_content(message.id)
    assert reset.content == ""
    assert reset.status == "streaming"

    store.append_stream_chunk(message.id, "42.")
    materialized = store.get_message(message.id)
    assert materialized.content == "42."


def test_reset_stream_content_noops_on_already_stopped_message():
    """Plan-B final-review LOW-1 (task-227): reset_stream_content must not
    resurrect an already-stopped message back to "streaming" -- mirrors
    append_stream_chunk's hardening for the same stop/cancel race family
    (Plan-B agent-runtime gate Finding 1). A disobedient model's
    post-stop tool-call turn calls reset_stream_content once its (leaked,
    already-dropped) turn is classified as a tool call; that must be a
    no-op once the user has already stopped the message, not leave it
    stuck "streaming" forever."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    store.append_stream_chunk(message.id, "before stop")
    stopped = store.mark_message_stopped(message.id)
    assert stopped.status == "stopped"
    assert stopped.content == "before stop"

    result = store.reset_stream_content(message.id)

    assert result.status == "stopped"
    assert result.content == "before stop"
    unchanged = store.get_message(message.id)
    assert unchanged.status == "stopped"
    assert unchanged.content == "before stop"


def test_store_tracks_active_workspace_context():
    context = ConsoleWorkspaceContext(active_workspace_id="workspace-a")
    store = ConsoleChatStore(workspace_context=context)

    assert store.workspace_context.active_workspace_id == "workspace-a"

    store.set_workspace_context(
        ConsoleWorkspaceContext(active_workspace_id="workspace-b")
    )

    assert store.workspace_context.active_workspace_id == "workspace-b"


def test_store_creates_and_switches_sessions():
    store = ConsoleChatStore()
    first = store.ensure_session(title="Chat 1")
    store.append_message(first.id, role=ConsoleMessageRole.USER, content="first")
    second = store.create_session(title="Chat 2")

    assert store.active_session_id == second.id

    store.switch_session(first.id)

    assert store.active_session_id == first.id
    assert store.messages_for_session(first.id)[0].content == "first"


def test_store_restore_state_replaces_sessions_and_rebuilds_message_indexes():
    store = ConsoleChatStore()
    stale_session = store.ensure_session(title="Stale")
    stale_message = store.append_message(
        stale_session.id,
        role=ConsoleMessageRole.USER,
        content="stale",
    )
    restored_session = ConsoleChatSession(id="session-a", title="Restored")
    restored_message = ConsoleChatMessage(
        id="message-a",
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
    )

    store.restore_state(
        sessions=[restored_session],
        messages_by_session={"session-a": [restored_message]},
        active_session_id="session-a",
    )

    assert [session.id for session in store.sessions()] == ["session-a"]
    assert store.active_session_id == "session-a"
    assert store.messages_for_session("session-a")[0].content == "answer"
    assert store.session_id_for_message("message-a") == "session-a"
    with pytest.raises(KeyError):
        store.get_message(stale_message.id)


def test_store_renames_session_with_trimmed_title():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")

    renamed, persisted = store.rename_session(session.id, "  Planning tab  ")

    assert renamed is session
    assert persisted is True
    assert store.sessions()[0].title == "Planning tab"


def test_store_rejects_blank_session_title_without_mutating_existing_title():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")

    with pytest.raises(ValueError):
        store.rename_session(session.id, "   ")

    assert store.sessions()[0].title == "Chat 1"


def test_console_sessions_store_independent_settings_snapshots() -> None:
    store = ConsoleChatStore()
    first_settings = ConsoleSessionSettings(
        provider="llama_cpp", model="a", temperature=0.1
    )
    second_settings = ConsoleSessionSettings(
        provider="openai", model="b", temperature=0.9
    )

    first = store.create_session(title="A", settings=first_settings)
    second = store.create_session(title="B", settings=second_settings)

    assert store.session_settings(first.id).model == "a"
    assert store.session_settings(second.id).model == "b"


def test_replacing_session_settings_does_not_mutate_other_sessions() -> None:
    store = ConsoleChatStore()
    first = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="a")
    )
    second = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="b")
    )

    store.replace_session_settings(
        first.id,
        ConsoleSessionSettings(provider="llama_cpp", model="changed"),
    )

    assert store.session_settings(first.id).model == "changed"
    assert store.session_settings(second.id).model == "b"


def test_replace_session_settings_returns_stored_session_instance() -> None:
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="a")
    )

    returned = store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(provider="llama_cpp", model="changed"),
    )

    assert returned is store.switch_session(session.id)
    assert returned.settings.model == "changed"


def test_ensure_session_applies_settings_only_when_creating_session() -> None:
    store = ConsoleChatStore()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="new")

    session = store.ensure_session(settings=settings)

    assert store.session_settings(session.id) == settings


def test_ensure_session_settings_do_not_mutate_existing_active_session() -> None:
    store = ConsoleChatStore()
    original_settings = ConsoleSessionSettings(provider="llama_cpp", model="original")
    session = store.ensure_session(settings=original_settings)

    ensured = store.ensure_session(
        settings=ConsoleSessionSettings(provider="openai", model="ignored"),
    )

    assert ensured.id == session.id
    assert store.session_settings(session.id) == original_settings


def test_session_settings_returns_none_when_session_has_no_settings() -> None:
    store = ConsoleChatStore()
    session = store.create_session()

    assert store.session_settings(session.id) is None


def test_store_closes_session_and_activates_neighbor():
    store = ConsoleChatStore()
    first = store.ensure_session(title="Chat 1")
    second = store.create_session(title="Chat 2")
    store.append_message(second.id, role=ConsoleMessageRole.USER, content="second")

    activated = store.close_session(second.id)

    assert activated == first
    assert store.active_session_id == first.id
    assert [session.id for session in store.sessions()] == [first.id]
    with pytest.raises(KeyError):
        store.messages_for_session(second.id)


def test_store_closes_last_session_returns_none():
    store = ConsoleChatStore()
    only = store.ensure_session(title="Solo")
    store.append_message(only.id, role=ConsoleMessageRole.USER, content="msg")

    activated = store.close_session(only.id)

    assert activated is None
    assert store.active_session_id is None
    assert store.sessions() == []


def test_store_adds_regenerated_variant_and_selects_it():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
    )

    store.add_variant(message.id, "second")

    updated = store.get_message(message.id)
    assert updated.variants.current.content == "second"
    assert updated.variants.can_go_previous is True


class FakePersistence:
    def __init__(self):
        self.created_conversations = []
        self.created_messages = []
        self.updated_messages = []
        self.updated_system_prompts = []
        self.updated_pinned_prefills = []
        self.roleplay_updates = []
        self.speech_updates = []
        self.conversation_version = 1
        self.speech_update_result = True
        self.restored_speech_preferences = None
        self.last_create_kwargs = None

    def create_conversation(self, **kwargs):
        self.created_conversations.append(kwargs)
        self.last_create_kwargs = kwargs
        return "conv-1"

    def promote_console_conversation_bundle(
        self,
        *,
        conversation_id,
        policy_candidate,
        conversation_kwargs,
        messages,
        active_leaf_message_id,
        context_summary=None,
        context_summary_boundary_message_id=None,
        contributions=(),
    ):
        if contributions:
            raise RuntimeError("FakePersistence does not execute contributions")
        self.created_conversations.append(
            {"conversation_id": conversation_id, **dict(conversation_kwargs)}
        )
        self.last_create_kwargs = dict(conversation_kwargs)
        for prepared in messages:
            self.created_messages.append(dict(prepared["create_kwargs"]))
        return ConsoleLibraryPolicySnapshot(
            auto_retrieve=policy_candidate.auto_retrieve,
            assistant_access=policy_candidate.assistant_access,
            policy_revision=1,
            source="durable",
        )

    def update_conversation_system_prompt(self, *, conversation_id, system_prompt):
        self.updated_system_prompts.append(
            {"conversation_id": conversation_id, "system_prompt": system_prompt}
        )
        return True

    def update_conversation_pinned_prefill(self, *, conversation_id, pinned_prefill):
        self.updated_pinned_prefills.append((conversation_id, pinned_prefill))
        return True

    def update_conversation_roleplay_context(
        self,
        *,
        conversation_id,
        user_name_override,
        character_system_template,
        character_name_snapshot,
    ):
        self.roleplay_updates.append(
            {
                "conversation_id": conversation_id,
                "user_name_override": user_name_override,
                "character_system_template": character_system_template,
                "character_name_snapshot": character_name_snapshot,
            }
        )
        return True

    def get_conversation_version(self, conversation_id):
        return self.conversation_version

    def update_conversation_speech_preferences(
        self, *, conversation_id, preferences, expected_version
    ):
        self.speech_updates.append(
            {
                "conversation_id": conversation_id,
                "preferences": preferences,
                "expected_version": expected_version,
            }
        )
        return self.speech_update_result

    def get_conversation_speech_preferences(self, conversation_id):
        return self.restored_speech_preferences

    def create_message(
        self,
        *,
        conversation_id,
        sender,
        content,
        image_data,
        image_mime_type,
        message_id=None,
        parent_message_id=None,
        feedback=None,
        metadata_json=None,
    ):
        kwargs = {
            "conversation_id": conversation_id,
            "sender": sender,
            "content": content,
            "image_data": image_data,
            "image_mime_type": image_mime_type,
            "message_id": message_id,
            "parent_message_id": parent_message_id,
            "feedback": feedback,
            "metadata_json": metadata_json,
        }
        self.created_messages.append(kwargs)
        return f"msg-{len(self.created_messages)}"

    def update_message_content(
        self,
        *,
        message_id,
        content,
        image_data,
        image_mime_type,
        parent_message_id=None,
        feedback=None,
        update_parent=False,
        update_feedback=False,
        metadata_json=None,
    ):
        self.updated_messages.append(
            {
                "message_id": message_id,
                "content": content,
                "image_data": image_data,
                "image_mime_type": image_mime_type,
                "parent_message_id": parent_message_id,
                "feedback": feedback,
                "update_parent": update_parent,
                "update_feedback": update_feedback,
                "metadata_json": metadata_json,
            }
        )
        return True


class FakeChatSyncProducer:
    def __init__(self):
        self.enqueued = []

    def enqueue_chat_message(self, **kwargs):
        self.enqueued.append(kwargs)
        return {
            "status": "enqueued",
            "outbox_entry": {
                "outbox_id": len(self.enqueued),
                "envelope": {
                    "payload_hash": f"hash:{kwargs['role']}:{kwargs['content']}",
                },
            },
        }


class FailingChatSyncProducer:
    def enqueue_chat_message(self, **kwargs):
        raise RuntimeError("sync unavailable")


def test_new_session_defaults_reply_speech_off():
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    session = ConsoleChatStore().ensure_session()

    assert session.speech_preferences == ConsoleSpeechPreferences()


def test_unsaved_session_stages_all_reply_speech_preferences():
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    store = ConsoleChatStore()
    session = store.ensure_session()
    destination = "sha256:" + "a" * 64

    assert store.set_auto_speak(session.id, True) == (session, True)
    assert store.pause_auto_speak(session.id) == (session, True)
    assert store.confirm_auto_speak_destination(session.id, destination) == (
        session,
        True,
    )
    assert session.speech_preferences == ConsoleSpeechPreferences(
        auto_speak=True,
        paused=True,
        consent_destination=destination,
    )
    assert store.resume_auto_speak(session.id) == (session, True)
    assert session.speech_preferences.paused is False


def test_reply_speech_preference_epoch_advances_only_after_successful_mutation() -> None:
    store = ConsoleChatStore()
    session = store.ensure_session()

    assert store.speech_preference_epoch(session.id) == 0
    store.set_auto_speak(session.id, True)
    assert store.speech_preference_epoch(session.id) == 1
    store.set_auto_speak(session.id, True)
    assert store.speech_preference_epoch(session.id) == 1
    store.confirm_auto_speak_destination(session.id, "sha256:" + "a" * 64)
    assert store.speech_preference_epoch(session.id) == 2
    store.set_auto_speak(session.id, False)
    store.set_auto_speak(session.id, True)
    assert store.speech_preference_epoch(session.id) == 4


def test_active_session_epoch_advances_across_a_b_a_and_restore() -> None:
    store = ConsoleChatStore()
    first = store.create_session(title="A")
    created_a = store.active_session_epoch()
    second = store.create_session(title="B")
    created_b = store.active_session_epoch()

    store.switch_session(first.id)
    switched_a = store.active_session_epoch()
    store.switch_session(second.id)
    switched_b = store.active_session_epoch()
    store.switch_session(first.id)
    returned_a = store.active_session_epoch()
    store.restore_state(sessions=[replace(first)], active_session_id=first.id)
    restored_a = store.active_session_epoch()

    assert created_a < created_b < switched_a < switched_b < returned_a < restored_a


def test_restore_state_replacement_advances_speech_epoch_monotonically() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    store.set_auto_speak(session.id, True)
    before_restore = store.speech_preference_epoch(session.id)
    restored = replace(session)

    store.restore_state(sessions=[restored], active_session_id=session.id)
    first_restore = store.speech_preference_epoch(session.id)
    store.restore_state(sessions=[restored], active_session_id=session.id)
    second_restore = store.speech_preference_epoch(session.id)

    assert before_restore < first_restore < second_restore


def test_failed_reply_speech_preference_write_does_not_advance_epoch() -> None:
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    session.persisted_conversation_id = "conv-1"
    persistence.update_conversation_speech_preferences = lambda **_kwargs: False

    _session, persisted = store.set_auto_speak(session.id, True)

    assert persisted is False
    assert store.speech_preference_epoch(session.id) == 0


def test_persisted_reply_speech_mutation_updates_memory_only_after_versioned_write():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    session.persisted_conversation_id = "conv-1"
    observed_before_write = []

    def write(**kwargs):
        observed_before_write.append(session.speech_preferences.auto_speak)
        persistence.speech_updates.append(kwargs)
        return True

    persistence.update_conversation_speech_preferences = write

    updated, persisted = store.set_auto_speak(session.id, True)

    assert updated is session
    assert persisted is True
    assert observed_before_write == [False]
    assert session.speech_preferences.auto_speak is True
    assert persistence.speech_updates[0]["expected_version"] == 1


@pytest.mark.parametrize("failure", [False, RuntimeError("write failed")])
def test_persisted_reply_speech_conflict_or_failure_is_nonmutating(failure):
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    session.persisted_conversation_id = "conv-1"
    before = session.speech_preferences

    def write(**kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    persistence.update_conversation_speech_preferences = write

    updated, persisted = store.set_auto_speak(session.id, True)

    assert updated is session
    assert persisted is False
    assert session.speech_preferences is before


def test_persisted_reply_speech_missing_version_is_nonmutating():
    persistence = FakePersistence()
    persistence.conversation_version = None
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    session.persisted_conversation_id = "conv-1"
    before = session.speech_preferences

    updated, persisted = store.pause_auto_speak(session.id)

    assert updated is session
    assert persisted is False
    assert session.speech_preferences is before
    assert persistence.speech_updates == []


def test_persisted_reply_speech_noop_reconciles_external_durable_change(tmp_path):
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    db = CharactersRAGDB(tmp_path / "speech-stale-noop.db", "speech-test")
    try:
        service = ChatPersistenceService(db)
        conversation_id = service.create_conversation(conversation_title="Saved")
        store = ConsoleChatStore(persistence=service)
        session = store.restore_persisted_session(
            title="Saved",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
        )
        assert session.speech_preferences == ConsoleSpeechPreferences()

        assert service.update_conversation_speech_preferences(
            conversation_id=conversation_id,
            preferences=ConsoleSpeechPreferences(auto_speak=True),
            expected_version=1,
        )

        updated, persisted = store.set_auto_speak(session.id, False)

        assert updated is session
        assert persisted is True
        assert session.speech_preferences == ConsoleSpeechPreferences()
        assert service.get_conversation_speech_preferences(
            conversation_id
        ) == ConsoleSpeechPreferences()
        assert db.get_conversation_by_id(conversation_id)["version"] == 3
    finally:
        db.close_connection()


def test_first_persist_includes_staged_reply_speech_preferences():
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    store.set_auto_speak(session.id, True)

    store.persist_session_if_needed(session.id)

    assert persistence.created_conversations[0]["speech_preferences"] == (
        ConsoleSpeechPreferences(auto_speak=True)
    )


def test_restore_persisted_session_round_trips_reply_speech_preferences():
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    persistence = FakePersistence()
    persistence.restored_speech_preferences = ConsoleSpeechPreferences(
        auto_speak=True,
        paused=True,
        consent_destination="sha256:" + "b" * 64,
    )
    store = ConsoleChatStore(persistence=persistence)

    session = store.restore_persisted_session(
        title="Saved",
        workspace_id=None,
        persisted_conversation_id="conv-1",
        all_nodes=[],
    )

    assert session.speech_preferences == persistence.restored_speech_preferences


def test_real_persistence_round_trips_roleplay_and_reply_speech_metadata(tmp_path):
    from tldw_chatbook.Chat.console_roleplay_metadata import (
        parse_console_roleplay_context,
    )

    db = CharactersRAGDB(tmp_path / "speech-preferences.db", "speech-test")
    try:
        service = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=service)
        session = store.create_session()
        session.user_display_name_override = "Rowan"
        store.set_auto_speak(session.id, True)
        store.confirm_auto_speak_destination(session.id, "sha256:" + "c" * 64)

        conversation_id = store.persist_session_if_needed(session.id)
        record = db.get_conversation_by_id(conversation_id)
        restored = ConsoleChatStore(persistence=service).restore_persisted_session(
            title="Saved",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
        )

        assert restored.speech_preferences == session.speech_preferences
        assert parse_console_roleplay_context(record["metadata"]).user_name_override == (
            "Rowan"
        )
    finally:
        db.close_connection()


def test_initial_reply_speech_is_inserted_at_version_one_with_sibling_metadata(
    tmp_path,
):
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    db = CharactersRAGDB(tmp_path / "speech-create-metadata.db", "speech-test")
    try:
        service = ChatPersistenceService(db)
        roleplay = {
            "version": 1,
            "user_name_override": "Rowan",
        }

        conversation_id = service.create_conversation(
            conversation_title="Speech setup",
            metadata={
                "console_roleplay_context": roleplay,
                "other": {"keep": True},
            },
            speech_preferences=ConsoleSpeechPreferences(auto_speak=True),
        )

        record = db.get_conversation_by_id(conversation_id)
        metadata = json.loads(record["metadata"])
        assert record["version"] == 1
        assert metadata["console_roleplay_context"] == roleplay
        assert metadata["other"] == {"keep": True}
        assert metadata["console_speech"]["auto_speak"] is True
        events = db.execute_query(
            "SELECT operation, payload FROM sync_log "
            "WHERE entity = 'conversations' AND entity_id = ?",
            (conversation_id,),
        ).fetchall()
        assert len(events) == 1
        assert events[0]["operation"] == "create"
        assert json.loads(events[0]["payload"])["metadata"] == record["metadata"]
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param("{", id="malformed-json"),
        pytest.param("[]", id="array-json"),
        pytest.param('"scalar"', id="string-json"),
        pytest.param("null", id="null-json"),
        pytest.param('{"bad": NaN}', id="non-finite-json"),
        pytest.param(["unsupported"], id="unsupported-type"),
        pytest.param({"bad": {"not-json"}}, id="unserializable-mapping"),
        pytest.param({"bad": float("nan")}, id="non-finite-mapping"),
        pytest.param({1: "coerced-key"}, id="non-string-mapping-key"),
    ],
)
@pytest.mark.parametrize("with_speech", [False, True])
def test_create_conversation_rejects_non_object_metadata_before_db_add(
    tmp_path,
    monkeypatch,
    metadata,
    with_speech,
):
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    db = CharactersRAGDB(
        tmp_path / f"speech-invalid-service-{with_speech}.db",
        "speech-test",
    )
    try:
        service = ChatPersistenceService(db)
        add_calls = []
        original_add = db.add_conversation

        def recording_add(conversation_data):
            add_calls.append(conversation_data)
            return original_add(conversation_data)

        monkeypatch.setattr(db, "add_conversation", recording_add)
        before_rows = db.execute_query(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        before_events = db.execute_query(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
        ).fetchone()[0]
        speech_preferences = (
            ConsoleSpeechPreferences(auto_speak=True) if with_speech else None
        )

        with pytest.raises(ValueError, match="metadata.*JSON object"):
            service.create_conversation(
                conversation_title="Invalid metadata",
                metadata=metadata,
                speech_preferences=speech_preferences,
            )

        assert add_calls == []
        assert (
            db.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == before_rows
        )
        assert (
            db.execute_query(
                "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
            ).fetchone()[0]
            == before_events
        )
    finally:
        db.close_connection()


def test_service_metadata_rejection_is_nonmutating_in_caller_transaction(tmp_path):
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    db = CharactersRAGDB(tmp_path / "speech-invalid-service-outer.db", "speech-test")
    try:
        service = ChatPersistenceService(db)
        connection = db.get_connection()
        before_rows = connection.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        before_events = connection.execute(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
        ).fetchone()[0]
        connection.execute("BEGIN")

        with pytest.raises(ValueError, match="metadata.*JSON object"):
            service.create_conversation(
                conversation_title="Invalid metadata",
                metadata="not-json",
                speech_preferences=ConsoleSpeechPreferences(auto_speak=True),
            )

        connection.commit()
        assert connection.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0] == before_rows
        assert connection.execute(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
        ).fetchone()[0] == before_events
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param("{", id="malformed-json"),
        pytest.param("[]", id="array-json"),
        pytest.param("1", id="number-json"),
        pytest.param("null", id="null-json"),
        pytest.param('{"bad": NaN}', id="non-finite-json"),
        pytest.param({"unsupported": True}, id="mapping-type"),
        pytest.param(["unsupported"], id="list-type"),
        pytest.param(1, id="number-type"),
    ],
)
def test_add_conversation_rejects_non_object_metadata_without_writes(
    tmp_path,
    metadata,
):
    db = CharactersRAGDB(tmp_path / "speech-invalid-db.db", "speech-test")
    try:
        before_rows = db.execute_query(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        before_events = db.execute_query(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
        ).fetchone()[0]

        with pytest.raises(InputError, match="metadata.*JSON object"):
            db.add_conversation({"title": "Invalid metadata", "metadata": metadata})

        assert (
            db.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == before_rows
        )
        assert (
            db.execute_query(
                "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
            ).fetchone()[0]
            == before_events
        )
    finally:
        db.close_connection()


def test_direct_metadata_rejection_is_nonmutating_in_caller_transaction(tmp_path):
    db = CharactersRAGDB(tmp_path / "speech-invalid-db-outer.db", "speech-test")
    try:
        connection = db.get_connection()
        before_rows = connection.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        before_events = connection.execute(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
        ).fetchone()[0]
        connection.execute("BEGIN")

        with pytest.raises(InputError, match="metadata.*JSON object"):
            db.add_conversation({"title": "Invalid metadata", "metadata": "[]"})

        connection.commit()
        assert connection.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0] == before_rows
        assert connection.execute(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
        ).fetchone()[0] == before_events
    finally:
        db.close_connection()


def test_add_conversation_accepts_object_metadata_in_one_create_event(tmp_path):
    db = CharactersRAGDB(tmp_path / "speech-valid-db.db", "speech-test")
    try:
        expected = {"other": {"keep": True}}

        conversation_id = db.add_conversation(
            {
                "title": "Valid metadata",
                "metadata": json.dumps(expected),
            }
        )

        record = db.get_conversation_by_id(conversation_id)
        events = db.execute_query(
            "SELECT operation, payload FROM sync_log "
            "WHERE entity = 'conversations' AND entity_id = ?",
            (conversation_id,),
        ).fetchall()
        assert record["version"] == 1
        assert json.loads(record["metadata"]) == expected
        assert len(events) == 1
        assert events[0]["operation"] == "create"
        assert json.loads(events[0]["payload"])["metadata"] == record["metadata"]
    finally:
        db.close_connection()


@pytest.mark.parametrize("caller_owned_transaction", [False, True])
def test_initial_future_speech_metadata_failure_never_creates_a_row(
    tmp_path,
    caller_owned_transaction,
):
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    db = CharactersRAGDB(
        tmp_path / f"speech-create-failure-{caller_owned_transaction}.db",
        "speech-test",
    )
    try:
        service = ChatPersistenceService(db)
        before = db.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
        connection = db.get_connection()
        if caller_owned_transaction:
            connection.execute("BEGIN")

        with pytest.raises(ValueError, match="version 2"):
            service.create_conversation(
                conversation_title="Failed speech setup",
                metadata={
                    "console_speech": {
                        "auto_speak": True,
                        "paused": False,
                        "consent_destination": None,
                        "consent_version": 2,
                    }
                },
                speech_preferences=ConsoleSpeechPreferences(auto_speak=True),
            )

        if caller_owned_transaction:
            connection.commit()
        after = db.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
        assert after == before
    finally:
        db.close_connection()


def test_future_speech_metadata_blocks_store_mutation_without_state_change(tmp_path):
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    db = CharactersRAGDB(tmp_path / "speech-future-version.db", "speech-test")
    try:
        service = ChatPersistenceService(db)
        conversation_id = service.create_conversation(conversation_title="Future")
        future_metadata = {
            "console_speech": {
                "auto_speak": True,
                "paused": False,
                "consent_destination": None,
                "consent_version": 2,
                "future_flag": "keep",
            }
        }
        assert db.update_conversation(
            conversation_id,
            {"metadata": json.dumps(future_metadata, sort_keys=True)},
            expected_version=1,
        )
        before_record = db.get_conversation_by_id(conversation_id)
        store = ConsoleChatStore(persistence=service)
        session = store.restore_persisted_session(
            title="Future",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
        )
        before_preferences = session.speech_preferences

        updated, persisted = store.set_auto_speak(session.id, True)

        after_record = db.get_conversation_by_id(conversation_id)
        assert updated is session
        assert persisted is False
        assert session.speech_preferences is before_preferences
        assert after_record["version"] == before_record["version"]
        assert json.loads(after_record["metadata"]) == future_metadata
        assert service.get_conversation_speech_preferences(
            conversation_id
        ) == ConsoleSpeechPreferences()
    finally:
        db.close_connection()


def test_store_can_persist_user_and_assistant_messages_through_adapter():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    store.persist_session_if_needed(session.id)
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hello", persist=True
    )

    assert persistence.created_conversations[0]["conversation_title"] == "Chat 1"
    assert persistence.created_messages[0]["conversation_id"] == "conv-1"
    assert persistence.created_messages[0]["sender"] == "user"
    assert persistence.created_messages[0]["content"] == "hello"
    assert persistence.created_messages[0]["image_data"] is None
    assert persistence.created_messages[0]["image_mime_type"] is None


def test_durable_resume_starts_with_a_fresh_empty_todo_store():
    """Session tasks are process-navigation state, not durable Chat data."""
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    live = store.ensure_session(title="Chat with tasks")
    live.todo_store.create(content="private-task-record-one")
    live.todo_store.create(content="private-task-record-deleted")
    live.todo_store.create(content="private-task-record-three")
    live.todo_store.update(task_id="2", expected_version=1, status="deleted")
    assert live.todo_store.export_snapshot()["next_id"] == 4

    conversation_id = store.persist_session_if_needed(live.id)

    durable_kwargs = persistence.created_conversations[0]
    assert persistence.last_create_kwargs == durable_kwargs
    durable_projection = repr(durable_kwargs)
    for forbidden in (
        "todo_state",
        "next_id",
        "private-task-record-one",
        "private-task-record-deleted",
        "private-task-record-three",
    ):
        assert forbidden not in durable_projection

    restored = store.restore_persisted_session(
        title=live.title,
        workspace_id=live.workspace_id,
        persisted_conversation_id=conversation_id,
        all_nodes=[],
    )

    assert restored.todo_store.list_after(None) == []
    assert restored.todo_store.create(content="Fresh after restart")["id"] == "1"


def test_persist_session_if_needed_passes_system_prompt_from_settings():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(
            provider="llama_cpp", system_prompt="Be terse."
        ),
    )

    store.persist_session_if_needed(session.id)

    assert persistence.created_conversations[0]["system_prompt"] == "Be terse."


def test_persist_session_if_needed_passes_none_system_prompt_without_settings():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    store.persist_session_if_needed(session.id)

    assert persistence.created_conversations[0]["system_prompt"] is None


def test_persist_session_if_needed_reports_invalid_runtime_backend():
    """Invalid provenance must fail visibly without any durable writes."""
    from loguru import logger as loguru_logger

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Malformed chat", runtime_backend="invalid")
    diagnostics: list[str] = []
    sink_id = loguru_logger.add(
        diagnostics.append,
        level="ERROR",
        format="{extra[session_id]} {extra[runtime_backend]} {message}",
    )
    try:
        with pytest.raises(
            ValueError, match="runtime_backend must be 'local' or 'server'"
        ):
            store.persist_session_if_needed(session.id)
        with pytest.raises(
            ValueError, match="runtime_backend must be 'local' or 'server'"
        ):
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="Keep this message in memory",
                persist=True,
            )
    finally:
        loguru_logger.remove(sink_id)

    assert persistence.created_conversations == []
    assert persistence.created_messages == []
    assert any(
        session.id in diagnostic
        and "'invalid'" in diagnostic
        and "persist" in diagnostic.lower()
        for diagnostic in diagnostics
    ), diagnostics


def test_persist_session_if_needed_handles_invalid_backend_with_raising_repr():
    """Diagnostic formatting cannot replace the stable invalid-backend error."""
    from loguru import logger as loguru_logger

    class ExplodingBackend:
        def __repr__(self):
            raise RuntimeError("repr must not run")

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(
        title="Malformed chat", runtime_backend=ExplodingBackend()
    )
    diagnostics: list[str] = []
    sink_id = loguru_logger.add(
        diagnostics.append,
        level="ERROR",
        format="{extra[session_id]} {extra[runtime_backend]} {message}",
    )
    try:
        with pytest.raises(
            ValueError, match="runtime_backend must be 'local' or 'server'"
        ):
            store.persist_session_if_needed(session.id)
    finally:
        loguru_logger.remove(sink_id)

    assert persistence.created_conversations == []
    assert persistence.created_messages == []
    assert any(
        session.id in diagnostic
        and "ExplodingBackend" in diagnostic
        and "persist" in diagnostic.lower()
        for diagnostic in diagnostics
    ), diagnostics


@pytest.mark.parametrize(
    ("session_kwargs", "expected"),
    [
        (
            {
                "runtime_backend": "local",
                "assistant_kind": "character",
                "assistant_id": "7",
                "assistant_authority_id": "local-authority",
                "character_id": 7,
                "character_name": "Elara",
            },
            {
                "runtime_backend": "local",
                "assistant_kind": "character",
                "assistant_id": "7",
                "assistant_authority_id": "local-authority",
                "character_id": 7,
                "character_name": "Elara",
            },
        ),
        (
            {
                "runtime_backend": "server",
                "assistant_kind": "character",
                "assistant_id": "opaque-character",
                "assistant_authority_id": "server-user-v1:" + ("b" * 64),
            },
            {
                "runtime_backend": "server",
                "assistant_kind": "character",
                "assistant_id": "opaque-character",
                "assistant_authority_id": "server-user-v1:" + ("b" * 64),
                "character_id": None,
                "character_name": None,
            },
        ),
        (
            {
                "runtime_backend": "server",
                "assistant_kind": "character",
                "assistant_id": "opaque-character",
                "assistant_authority_id": None,
            },
            {
                "runtime_backend": "server",
                "assistant_kind": "character",
                "assistant_id": "opaque-character",
                "assistant_authority_id": None,
                "character_id": None,
                "character_name": None,
            },
        ),
        (
            {
                "runtime_backend": "local",
                "assistant_kind": "persona",
                "assistant_id": "persona-1",
                "assistant_authority_id": None,
                "character_id": 99,
            },
            {
                "runtime_backend": "local",
                "assistant_kind": "persona",
                "assistant_id": "persona-1",
                "assistant_authority_id": None,
                "character_id": None,
                "character_name": None,
            },
        ),
        (
            {
                "runtime_backend": "local",
                "assistant_kind": "character",
                "assistant_id": "007",
                "assistant_authority_id": "local-authority",
                "character_id": 7,
                "character_name": "Mismatched",
            },
            {
                "runtime_backend": "local",
                "assistant_kind": "character",
                "assistant_id": "007",
                "assistant_authority_id": "local-authority",
                "character_id": None,
                "character_name": None,
            },
        ),
        (
            {
                "runtime_backend": "local",
                "assistant_kind": "character",
                "assistant_id": "0",
                "assistant_authority_id": "local-authority",
                "character_id": 0,
                "character_name": "Invalid",
            },
            {
                "runtime_backend": "local",
                "assistant_kind": "character",
                "assistant_id": "0",
                "assistant_authority_id": "local-authority",
                "character_id": None,
                "character_name": None,
            },
        ),
        (
            {},
            {
                "runtime_backend": "local",
                "assistant_kind": "generic",
                "assistant_id": "console",
                "assistant_authority_id": None,
                "character_id": None,
                "character_name": None,
            },
        ),
    ],
    ids=[
        "local-character",
        "scoped-server",
        "unscoped-server",
        "persona",
        "noncanonical-local-id",
        "nonpositive-local-id",
        "generic",
    ],
)
def test_persist_session_if_needed_passes_exact_session_identity(
    session_kwargs, expected
):
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Identity chat", **session_kwargs)

    conv_id = store.persist_session_if_needed(session.id)

    assert conv_id is not None
    kwargs = persistence.last_create_kwargs
    assert {key: kwargs[key] for key in expected} == expected


def test_persist_session_if_needed_non_character_stays_generic():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")

    store.persist_session_if_needed(session.id)

    kwargs = persistence.last_create_kwargs
    assert kwargs["runtime_backend"] == "local"
    assert kwargs["assistant_kind"] == "generic"
    assert kwargs["assistant_id"] == "console"
    assert kwargs["assistant_authority_id"] is None
    assert kwargs["character_id"] is None


def test_set_session_system_prompt_updates_settings_without_persisting_when_unsaved():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )

    updated, persisted = store.set_session_system_prompt(
        session.id, "New system prompt"
    )

    assert updated.settings.system_prompt == "New system prompt"
    assert persisted is True
    assert persistence.updated_system_prompts == []
    assert persistence.created_conversations == []


def test_set_session_system_prompt_persists_change_when_conversation_already_saved():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )
    store.persist_session_if_needed(session.id)

    updated, persisted = store.set_session_system_prompt(
        session.id, "Answer in French."
    )

    assert updated.settings.system_prompt == "Answer in French."
    assert persisted is True
    assert persistence.updated_system_prompts == [
        {"conversation_id": "conv-1", "system_prompt": "Answer in French."}
    ]


def test_set_session_system_prompt_preserves_formatting_verbatim():
    """Only blank/whitespace-only text is treated as "no system prompt";
    leading whitespace and internal formatting (e.g. a blank line between
    paragraphs) must survive into storage unchanged rather than being
    stripped."""
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )
    store.persist_session_if_needed(session.id)
    formatted_prompt = "  line1\n\n  line2  "

    updated, persisted = store.set_session_system_prompt(session.id, formatted_prompt)

    assert updated.settings.system_prompt == formatted_prompt
    assert persisted is True
    assert persistence.updated_system_prompts == [
        {"conversation_id": "conv-1", "system_prompt": formatted_prompt}
    ]


def test_set_session_system_prompt_normalizes_blank_to_none():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(
            provider="llama_cpp", system_prompt="Old prompt"
        ),
    )
    store.persist_session_if_needed(session.id)

    updated, persisted = store.set_session_system_prompt(session.id, "   ")

    assert updated.settings.system_prompt is None
    assert persisted is True
    assert persistence.updated_system_prompts == [
        {"conversation_id": "conv-1", "system_prompt": None}
    ]


def test_set_session_system_prompt_survives_persistence_failure_without_log_leak(
    caplog: pytest.LogCaptureFixture,
):
    """A persistence error (e.g. the conversation was deleted, or a DB
    conflict) must not escape `set_session_system_prompt`, and the
    in-memory session keeps the applied value (this store's existing
    convention: mutations are not rolled back when the durable write that
    follows them fails); the caller gets `persisted=False` back so it can
    surface the failure honestly instead of assuming the change was saved.
    """

    import logging

    from loguru import logger as loguru_logger

    system_sentinel = "TASK199_SYSTEM_BODY_MUST_NOT_LEAK"
    fingerprint_sentinel = "TASK199_SYSTEM_FINGERPRINT_MUST_NOT_LEAK"
    exception_sentinel = "TASK199_ADAPTER_EXCEPTION_MUST_NOT_LEAK"

    class RaisingPersistence(FakePersistence):
        def update_conversation_system_prompt(self, *, conversation_id, system_prompt):
            raise RuntimeError(
                f"{exception_sentinel}: system_prompt={system_prompt!r}; "
                f"fingerprint={fingerprint_sentinel}"
            )

    persistence = RaisingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )
    store.persist_session_if_needed(session.id)

    captured_logs: list[object] = []
    caplog.set_level(logging.ERROR, logger="task199.console_store")

    def capture_loguru(message: object) -> None:
        captured_logs.append(message)
        logging.getLogger("task199.console_store").error(str(message))

    sink_id = loguru_logger.add(capture_loguru, level="ERROR")
    try:
        updated, persisted = store.set_session_system_prompt(
            session.id,
            system_sentinel,
        )
    finally:
        loguru_logger.remove(sink_id)

    assert persisted is False
    assert updated.settings.system_prompt == system_sentinel
    assert store.session_settings(session.id).system_prompt == system_sentinel
    assert captured_logs
    rendered_logs = "\n".join(
        rendered
        for message in captured_logs
        for rendered in (str(message), repr(message))
    )
    for sentinel in (
        system_sentinel,
        fingerprint_sentinel,
        exception_sentinel,
    ):
        assert sentinel not in rendered_logs
        assert sentinel not in caplog.text
    assert "Traceback" not in rendered_logs
    assert "operation=set_session_system_prompt" in rendered_logs
    assert "context=durable_write" in rendered_logs
    assert "exception_category=RuntimeError" in rendered_logs


def test_set_session_pinned_prefill_updates_memory_and_writes_through():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")
    session.settings = ConsoleSessionSettings(provider="llama_cpp")
    session.persisted_conversation_id = "conv-1"

    updated, persisted = store.set_session_pinned_prefill(session.id, "Voice:")
    assert persisted is True
    assert updated.settings.pinned_prefill == "Voice:"
    assert persistence.updated_pinned_prefills == [("conv-1", "Voice:")]

    updated, persisted = store.set_session_pinned_prefill(session.id, None)
    assert updated.settings.pinned_prefill is None
    assert persistence.updated_pinned_prefills[-1] == ("conv-1", None)


def test_set_session_pinned_prefill_blank_normalizes_to_none():
    store = ConsoleChatStore()
    session = store.create_session(title="Chat 1")
    session.settings = ConsoleSessionSettings(provider="llama_cpp")
    updated, persisted = store.set_session_pinned_prefill(session.id, "   ")
    assert updated.settings.pinned_prefill is None
    assert persisted is True  # no durable write needed


def test_set_session_pinned_prefill_persistence_failure_keeps_memory():
    class ExplodingPersistence(FakePersistence):
        def update_conversation_pinned_prefill(self, **kwargs):
            raise RuntimeError("db locked")

    persistence = ExplodingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")
    session.settings = ConsoleSessionSettings(provider="llama_cpp")
    session.persisted_conversation_id = "conv-1"
    updated, persisted = store.set_session_pinned_prefill(session.id, "Voice:")
    assert persisted is False
    assert updated.settings.pinned_prefill == "Voice:"


def test_persist_session_if_needed_flushes_pinned_prefill():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")
    session.settings = ConsoleSessionSettings(
        provider="llama_cpp", pinned_prefill="Voice:"
    )
    store.persist_session_if_needed(session.id)
    assert persistence.updated_pinned_prefills == [("conv-1", "Voice:")]


def test_store_enqueues_chat_sync_after_user_message_is_durable():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
        sync_v2_authenticated_principal_id="user-a",
    )
    session = store.ensure_session(title="Chat 1")

    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        persist=True,
    )

    assert message.persisted_message_id == "msg-1"
    assert sync_producer.enqueued == [
        {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": None,
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "role": "user",
            "content": "hello",
            "parent_message_id": None,
            "sequence": 1,
            "variant_turn_id": None,
            "variant_index": None,
            "variant_count": None,
            "selected_variant_id": None,
            "base_version": None,
            "entity_version": None,
        }
    ]


def test_store_enqueues_streaming_assistant_only_after_completion():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello?",
        persist=True,
    )
    sync_producer.enqueued.clear()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )

    store.append_stream_chunk(assistant.id, "hel")
    store.append_stream_chunk(assistant.id, "lo")

    assert sync_producer.enqueued == []

    completed = store.mark_message_complete(assistant.id)

    assert completed.persisted_message_id == "msg-2"
    assert sync_producer.enqueued[-1]["message_id"] == "msg-2"
    assert sync_producer.enqueued[-1]["role"] == "assistant"
    assert sync_producer.enqueued[-1]["content"] == "hello"
    assert sync_producer.enqueued[-1]["parent_message_id"] == "msg-1"
    assert sync_producer.enqueued[-1]["sequence"] == 2


def test_store_does_not_enqueue_failed_assistant_final_content():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )

    store.append_stream_chunk(assistant.id, "partial")
    store.mark_message_failed(assistant.id)

    assert sync_producer.enqueued == []


def test_mark_message_failed_restores_prior_status_when_variant_base_present():
    """Plan-B Task 1 finding: a zero-chunk (empty-stream) regenerate of a
    previously-complete message must restore that prior status, not flip to
    "failed" -- every send path builds provider context with skip_failed=True
    (see console_chat_controller._provider_messages_for_session), so a wrong
    "failed" status here would silently drop an otherwise-good turn from the
    model's context for the rest of the session. Pre-refactor, a failed
    regenerate was a pure no-op on the existing message."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="original"
    )
    assert message.status == "complete"

    store.begin_variant_stream(message.id)
    # Zero-chunk stream: no append_stream_chunk calls before failure.
    failed = store.mark_message_failed(message.id)

    assert failed.status == "complete"
    assert failed.content == "original"
    assert message.id not in store._variant_stream_bases


def test_mark_message_failed_without_variant_base_still_marks_failed():
    """A normal (non-regenerate) send failure keeps today's "failed" status;
    only the variant-regenerate path has a known-good prior state to
    restore."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(assistant.id, "partial")

    failed = store.mark_message_failed(assistant.id)

    assert failed.status == "failed"
    assert failed.content == "partial"


def test_mark_message_send_blocked_fails_a_user_row_for_context_exclusion():
    """TASK-457(a): a USER row echoed before the readiness probe but rejected by
    the provider must be excludable from the next send's provider context. Unlike
    ``mark_message_failed`` (assistant-stream-only), this marks a never-streamed
    row failed with no terminal guard, and the flip lands on the stored row."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hello"
    )

    blocked = store.mark_message_send_blocked(user.id)

    assert blocked.status == "failed"
    assert blocked.content == "hello"
    assert blocked.role is ConsoleMessageRole.USER
    stored = store.messages_for_session(session.id)[0]
    assert stored.status == "failed"


def test_mark_message_send_blocked_rejects_non_user_rows():
    """TASK-457(a) (Qodo #777 review): the send-block path is for a never-
    streamed USER echo only. It must reject assistant/system rows so a mistaken
    caller cannot flip them to failed and bypass the assistant terminal-state
    guards (mark_message_failed's job)."""
    store = ConsoleChatStore()
    session = store.ensure_session()

    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    with pytest.raises(ValueError):
        store.mark_message_send_blocked(assistant.id)

    system = store.append_message(
        session.id, role=ConsoleMessageRole.SYSTEM, content="note"
    )
    with pytest.raises(ValueError):
        store.mark_message_send_blocked(system.id)


def test_persist_message_if_needed_flushes_a_deferred_message():
    """TASK-485: a message appended with persist=False stays out of the durable
    store until persist_message_if_needed flushes it (used on send-accept so a
    blocked attempt persists nothing); the flush creates the conversation and is
    idempotent."""
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    message = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hello", persist=False
    )
    assert persistence.created_messages == []
    assert persistence.created_conversations == []

    store.persist_message_if_needed(message.id)
    assert len(persistence.created_conversations) == 1
    assert len(persistence.created_messages) == 1
    assert persistence.created_messages[0]["content"] == "hello"

    # Idempotent — a second flush does not double-insert.
    store.persist_message_if_needed(message.id)
    assert len(persistence.created_messages) == 1


def test_store_persists_chat_when_sync_enqueue_fails():
    persistence = FakePersistence()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=FailingChatSyncProducer(),
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")

    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        persist=True,
    )

    assert message.persisted_message_id == "msg-1"
    assert persistence.created_messages[0]["content"] == "hello"


def test_store_enqueues_selected_variant_with_restore_metadata():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
        persist=True,
    )
    sync_producer.enqueued.clear()

    updated = store.add_variant(message.id, "second")

    assert updated.variants is not None
    assert sync_producer.enqueued[-1]["message_id"] == "msg-1"
    assert sync_producer.enqueued[-1]["content"] == "second"
    assert sync_producer.enqueued[-1]["base_version"] == "hash:assistant:first"
    assert sync_producer.enqueued[-1]["sequence"] == 1
    assert sync_producer.enqueued[-1]["variant_turn_id"] == updated.variants.turn_id
    assert sync_producer.enqueued[-1]["variant_index"] == 1
    assert sync_producer.enqueued[-1]["variant_count"] == 2
    assert (
        sync_producer.enqueued[-1]["selected_variant_id"] == updated.variants.current.id
    )


def test_store_sequences_only_sync_eligible_messages():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")
    store.append_message(
        session.id,
        role=ConsoleMessageRole.SYSTEM,
        content="visible only",
        persist=False,
    )
    failed = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(failed.id, "partial")
    store.mark_message_failed(failed.id)

    first_synced = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        persist=True,
    )
    second_synced = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="again",
        persist=True,
    )

    assert first_synced.persisted_message_id == "msg-2"
    assert second_synced.persisted_message_id == "msg-3"
    assert [entry["sequence"] for entry in sync_producer.enqueued] == [1, 2]


def test_store_updates_persisted_streaming_assistant_content_and_status():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    store.persist_session_if_needed(session.id)
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )

    store.append_stream_chunk(assistant.id, "hel")
    store.append_stream_chunk(assistant.id, "lo")
    store.mark_message_complete(assistant.id)

    completed = store.get_message(assistant.id)
    assert (
        persistence.updated_messages[-1]["message_id"] == completed.persisted_message_id
    )
    assert persistence.updated_messages[-1]["content"] == "hello"
    assert persistence.updated_messages[-1]["image_data"] is None
    assert persistence.updated_messages[-1]["image_mime_type"] is None


def test_store_persists_workspace_session_with_real_chat_persistence_service(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="test_client")
        )
        registry.create_workspace(workspace_id="workspace-a", name="Workspace A")
        store = ConsoleChatStore(
            persistence=ChatPersistenceService(db, workspace_registry=registry)
        )
        session = store.ensure_session(title="Chat 1", workspace_id="workspace-a")

        conversation_id = store.persist_session_if_needed(session.id)
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hello",
            persist=True,
        )

        conversation = db.get_conversation_by_id(conversation_id)
        persisted_message = db.get_message_by_id(message.persisted_message_id)
        assert conversation["scope_type"] == "workspace"
        assert conversation["workspace_id"] == "workspace-a"
        assert conversation["assistant_kind"] == "generic"
        assert conversation["assistant_id"] == "console"
        assert persisted_message["content"] == "hello"
        workspace_conversations = registry.list_workspace_conversations("workspace-a")
        assert [item.item_id for item in workspace_conversations] == [conversation_id]
    finally:
        db.close()


@pytest.mark.parametrize(
    "runtime_backend",
    (
        pytest.param("", id="missing"),
        pytest.param(123, id="non-string"),
        pytest.param("remote", id="unknown"),
    ),
)
def test_invalid_runtime_source_never_reaches_real_chat_persistence(
    tmp_path,
    monkeypatch,
    runtime_backend,
):
    """Malformed source provenance cannot become a durable local character."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        character_id = db.add_character_card({"name": "Existing local card"})
        assert type(character_id) is int
        character_count = db.count_character_cards()
        persistence = ChatPersistenceService(db)
        create_calls = []
        real_create = persistence.create_conversation

        def recording_create(**kwargs):
            create_calls.append(kwargs)
            return real_create(**kwargs)

        monkeypatch.setattr(persistence, "create_conversation", recording_create)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(
            title="Malformed character",
            runtime_backend=runtime_backend,
            assistant_kind="character",
            assistant_id=str(character_id),
            assistant_authority_id=db.get_local_authority_id(),
            character_id=character_id,
            character_name="Injected local card",
        )

        with pytest.raises(
            ValueError, match="runtime_backend must be 'local' or 'server'"
        ):
            store.persist_session_if_needed(session.id)
        with pytest.raises(
            ValueError, match="runtime_backend must be 'local' or 'server'"
        ):
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="Keep this message in memory",
                persist=True,
            )

        message = store.messages_for_session(session.id)[-1]
        assert session.persisted_conversation_id is None
        assert message.persisted_message_id is None
        assert create_calls == []
        assert db.get_all_conversation_ids() == []
        assert db.count_character_cards() == character_count
    finally:
        db.close()


def test_store_persists_default_workspace_chat_without_runtime_access(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="test_client")
        )
        registry.ensure_default_workspace()
        store = ConsoleChatStore(
            persistence=ChatPersistenceService(db, workspace_registry=registry)
        )
        session = store.ensure_session(
            title="Chat 1", workspace_id=DEFAULT_WORKSPACE_ID
        )

        conversation_id = store.persist_session_if_needed(session.id)
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="default workspace chat remains usable",
            persist=True,
        )

        conversation = db.get_conversation_by_id(conversation_id)
        persisted_message = db.get_message_by_id(message.persisted_message_id)
        workspace_conversations = registry.list_workspace_conversations(
            DEFAULT_WORKSPACE_ID
        )
        assert conversation is not None
        assert persisted_message is not None
        assert conversation["scope_type"] == "workspace"
        assert conversation["workspace_id"] == DEFAULT_WORKSPACE_ID
        assert persisted_message["content"] == "default workspace chat remains usable"
        assert [item.item_id for item in workspace_conversations] == [conversation_id]
        assert registry.list_runtime_bindings(DEFAULT_WORKSPACE_ID) == ()
    finally:
        db.close()


def test_store_system_prompt_round_trips_through_real_chat_persistence_service(
    tmp_path,
):
    """Persistence round-trip: create, apply a system prompt, reload from the real DB.

    Covers the Task 0 persistence seam end to end: creating a conversation
    with a session-level system prompt, then changing it once the
    conversation is already saved (the update path Task 0 flagged as
    missing), then reading the raw DB row back -- independent of any
    in-memory store state -- to confirm the change is truly durable.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.ensure_session(
            title="Chat 1",
            settings=ConsoleSessionSettings(
                provider="llama_cpp", system_prompt="Be terse."
            ),
        )

        conversation_id = store.persist_session_if_needed(session.id)
        assert (
            db.get_conversation_by_id(conversation_id)["system_prompt"] == "Be terse."
        )

        store.set_session_system_prompt(session.id, "Answer only in French.")

        # Read straight from the DB (not through the in-memory store) to
        # confirm the update is durable, the way a reload/reopen would see it.
        reloaded = db.get_conversation_by_id(conversation_id)
        assert reloaded["system_prompt"] == "Answer only in French."
        assert (
            store.session_settings(session.id).system_prompt == "Answer only in French."
        )
    finally:
        db.close()


def test_update_conversation_pinned_prefill_preserves_sibling_metadata(tmp_path):
    import json

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    service = ChatPersistenceService(db)
    conversation_id = service.create_conversation(
        assistant_kind="generic", assistant_id="console", conversation_title="T"
    )
    # Pre-seed a sibling key the dictionary-attach feature owns.
    record = db.get_conversation_by_id(conversation_id)
    db.update_conversation(
        conversation_id,
        {"metadata": json.dumps({"active_dictionaries": [1, 2]})},
        expected_version=record["version"],
    )

    assert service.update_conversation_pinned_prefill(
        conversation_id=conversation_id, pinned_prefill="Voice:"
    )
    meta = json.loads(db.get_conversation_by_id(conversation_id)["metadata"])
    assert meta["active_dictionaries"] == [1, 2]
    assert meta["pinned_response_prefill"] == "Voice:"

    assert service.update_conversation_pinned_prefill(
        conversation_id=conversation_id, pinned_prefill=None
    )
    meta = json.loads(db.get_conversation_by_id(conversation_id)["metadata"])
    assert meta["active_dictionaries"] == [1, 2]
    assert "pinned_response_prefill" not in meta

    assert not service.update_conversation_pinned_prefill(
        conversation_id="missing-conv", pinned_prefill="x"
    )


def test_store_delays_empty_assistant_persistence_until_terminal_content_with_real_service(
    tmp_path,
):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.ensure_session(title="Chat 1")
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )

        assert store.get_message(assistant.id).persisted_message_id is None

        store.append_stream_chunk(assistant.id, "hel")
        store.append_stream_chunk(assistant.id, "lo")
        completed = store.mark_message_complete(assistant.id)

        assert completed.persisted_message_id is not None
        persisted_message = db.get_message_by_id(completed.persisted_message_id)
        assert persisted_message["content"] == "hello"
    finally:
        db.close()


def test_store_rejects_streaming_chunks_for_non_assistant_message():
    store = ConsoleChatStore()
    session = store.ensure_session()
    user_message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
    )

    with pytest.raises(ValueError, match="Only assistant messages"):
        store.append_stream_chunk(user_message.id, "nope")


def test_store_rejects_streaming_chunks_after_terminal_state():
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.mark_message_failed(assistant.id)

    with pytest.raises(ValueError, match="Cannot append stream chunks"):
        store.append_stream_chunk(assistant.id, "late")


def test_store_drops_late_stream_chunks_for_stopped_message_silently():
    """Plan-B agent-runtime gate Finding 1 (stop-before-first-token race):
    a chunk that arrives after the message was already marked stopped must
    be dropped, not raise -- it's benign (the user already stopped this
    message), unlike a chunk arriving for a complete/failed message."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.append_stream_chunk(assistant.id, "before stop")
    stopped = store.mark_message_stopped(assistant.id)
    assert stopped.status == "stopped"
    assert stopped.content == "before stop"

    result = store.append_stream_chunk(assistant.id, "late chunk")

    assert result.status == "stopped"
    assert result.content == "before stop"
    unchanged = store.get_message(assistant.id)
    assert unchanged.status == "stopped"
    assert unchanged.content == "before stop"


def test_store_still_rejects_streaming_chunks_for_complete_message():
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.append_stream_chunk(assistant.id, "done text")
    store.mark_message_complete(assistant.id)

    with pytest.raises(ValueError, match="Cannot append stream chunks"):
        store.append_stream_chunk(assistant.id, "late")


def test_store_returns_message_snapshots_not_mutable_internals():
    store = ConsoleChatStore()
    session = store.ensure_session()
    user_message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
    )

    user_message.content = "external mutation"
    listed = store.messages_for_session(session.id)
    listed[0].status = "failed"

    stored = store.get_message(user_message.id)
    assert stored.content == "hello"
    assert stored.status == "complete"


def test_create_session_records_updated_at():
    store = ConsoleChatStore()
    session = store.create_session()
    parsed = datetime.fromisoformat(session.updated_at)
    assert parsed.tzinfo is not None


def test_append_message_touches_session_updated_at():
    store = ConsoleChatStore()
    session = store.create_session()
    original = session.updated_at
    store._sessions[session.id].updated_at = "2020-01-01T00:00:00+00:00"

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")

    touched = store._sessions[session.id].updated_at
    assert touched != "2020-01-01T00:00:00+00:00"
    assert datetime.fromisoformat(touched) >= datetime.fromisoformat(original)


from tldw_chatbook.Chat.attachment_core import PendingAttachment  # noqa: E402


def _image_attachment(name="photo.png"):
    return PendingAttachment(
        file_path=f"/tmp/{name}",
        display_name=name,
        file_type="image",
        insert_mode="attachment",
        data=b"\x89PNG-bytes",
        mime_type="image/png",
        original_size=11,
        processed_size=11,
    )


class RecordingPersistence:
    def __init__(self):
        self.created = []
        self.updated = []
        self._counter = 0

    def create_conversation(self, **kwargs):
        return "conv-1"

    def create_message(self, **kwargs):
        self.created.append(kwargs)
        self._counter += 1
        return f"msg-{self._counter}"

    def update_message_content(self, **kwargs):
        self.updated.append(kwargs)
        return True


def test_pending_attachment_is_per_session():
    store = ConsoleChatStore()
    first = store.create_session(title="A")
    second = store.create_session(title="B")

    store.set_pending_attachment(first.id, _image_attachment())

    assert store.pending_attachment(first.id) is not None
    assert store.pending_attachment(second.id) is None

    store.clear_pending_attachment(first.id)
    assert store.pending_attachment(first.id) is None


def test_append_message_persists_image_fields():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()

    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="what is this?",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        attachment_label="photo.png · 11 B",
        persist=True,
    )

    assert message.image_data == b"\x89PNG-bytes"
    assert message.attachment_label == "photo.png · 11 B"
    assert persistence.created[-1]["image_data"] == b"\x89PNG-bytes"
    assert persistence.created[-1]["image_mime_type"] == "image/png"


def test_image_only_user_message_persists_immediately():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()

    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        persist=True,
    )

    assert len(persistence.created) == 1
    assert persistence.created[0]["content"] == ""
    assert persistence.created[0]["image_data"] == b"\x89PNG-bytes"


def test_editing_message_content_does_not_wipe_persisted_image():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="original",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        persist=True,
    )

    store.update_message_content(message.id, "edited")

    assert persistence.updated[-1]["image_data"] == b"\x89PNG-bytes"
    assert persistence.updated[-1]["image_mime_type"] == "image/png"


from tldw_chatbook.Chat.console_chat_models import MessageAttachment  # noqa: E402
from tldw_chatbook.Chat.console_chat_store import MAX_PENDING_ATTACHMENTS  # noqa: E402


def _att(name="a.png", data=b"img", position=1):
    return MessageAttachment(
        data=data, mime_type="image/png", display_name=name, position=position
    )


def test_append_message_with_attachments_mirrors_first_into_scalars():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="look",
        attachments=(
            _att("a.png", b"img-1", 0),
            _att("b.jpg", b"img-2", 1),
        ),
    )
    assert len(message.attachments) == 2
    assert message.image_data == b"img-1"
    assert message.image_mime_type == "image/png"
    assert message.attachment_label and "a.png" in message.attachment_label


def test_append_message_scalar_kwargs_become_single_attachment():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=b"img",
        image_mime_type="image/png",
        attachment_label="pic.png · 3 B",
    )
    assert len(message.attachments) == 1
    assert message.attachments[0].data == b"img"
    assert message.image_data == b"img"


def test_pending_list_appends_caps_and_clears():
    store = ConsoleChatStore()
    session = store.ensure_session()
    from tldw_chatbook.Chat.attachment_core import PendingAttachment

    def _pending(name):
        return PendingAttachment(
            file_path=f"/tmp/{name}",
            display_name=name,
            file_type="image",
            insert_mode="attachment",
            data=b"x",
            mime_type="image/png",
            original_size=1,
            processed_size=1,
        )

    for index in range(MAX_PENDING_ATTACHMENTS):
        assert (
            store.add_pending_attachment(session.id, _pending(f"f{index}.png")) is True
        )
    assert store.add_pending_attachment(session.id, _pending("overflow.png")) is False
    assert len(store.pending_attachments(session.id)) == MAX_PENDING_ATTACHMENTS

    # Legacy single accessors still work over the list.
    assert store.pending_attachment(session.id).display_name == "f0.png"
    store.clear_pending_attachments(session.id)
    assert store.pending_attachments(session.id) == []
    assert store.pending_attachment(session.id) is None


def test_legacy_set_pending_attachment_replaces_all():
    store = ConsoleChatStore()
    session = store.ensure_session()
    from tldw_chatbook.Chat.attachment_core import PendingAttachment

    def _pending(name):
        return PendingAttachment(
            file_path=f"/tmp/{name}",
            display_name=name,
            file_type="image",
            insert_mode="attachment",
            data=b"x",
            mime_type="image/png",
            original_size=1,
            processed_size=1,
        )

    store.add_pending_attachment(session.id, _pending("a.png"))
    store.add_pending_attachment(session.id, _pending("b.png"))
    store.set_pending_attachment(session.id, _pending("only.png"))
    names = [p.display_name for p in store.pending_attachments(session.id)]
    assert names == ["only.png"]


# RecordingPersistence is defined in a pre-existing (origin/dev) region of
# this file, so it is subclassed here rather than edited. Its **kwargs-based
# create_message/update_message_content already record every kwarg the store
# sends, including the new ``attachments`` parameter.
class RecordingAttachmentPersistence(RecordingPersistence):
    pass  # create_message / update_message_content already record kwargs


def test_persist_new_message_sends_full_attachment_list():
    persistence = RecordingAttachmentPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="multi",
        attachments=(
            _att("a.png", b"img-0"),
            _att("b.png", b"img-1"),
        ),
        persist=True,
    )
    sent = persistence.created[-1]["attachments"]
    assert [a["position"] for a in sent] == [0, 1]
    assert sent[0]["data"] == b"img-0"
    assert sent[1]["display_name"] == "b.png"
    # The service derives the legacy image columns from position 0 when
    # attachments is provided, but create_message's image_data/
    # image_mime_type kwargs are keyword-only, so the store still sends
    # explicit None scalars alongside attachments (defense in depth; a P0
    # live crash was caused by omitting them against the real service,
    # which declared them required with no defaults).
    assert persistence.created[-1]["image_data"] is None
    assert persistence.created[-1]["image_mime_type"] is None


def test_persist_new_message_sends_data_bearing_attachments_only():
    persistence = RecordingAttachmentPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="multi",
        attachments=(
            _att("a.png", b"img-0"),
            _att("hollow.png", None),
            _att("c.png", b"img-2"),
        ),
        persist=True,
    )
    sent = persistence.created[-1]["attachments"]
    # The hollow (data=None) attachment is skipped; surviving entries keep
    # their re-based positions rather than being compacted.
    assert [a["position"] for a in sent] == [0, 2]
    assert [a["display_name"] for a in sent] == ["a.png", "c.png"]


def test_persist_edit_leaves_attachments_none():
    persistence = RecordingAttachmentPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="x",
        attachments=(_att("a.png", b"img"),),
        persist=True,
    )
    store.update_message_content(message.id, "edited")
    assert persistence.updated[-1]["attachments"] is None


# ---------------------------------------------------------------------------
# TASK-259: `_materialize_stream_buffer` collapses the chunk list after each
# join so a later materialize joins only chunks that arrived since. The
# invariant `"".join(buffer) == full streamed content` must hold throughout.
# ---------------------------------------------------------------------------


def test_materialize_collapses_stream_buffer_after_join():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    store.append_stream_chunk(message.id, "one ")
    store.append_stream_chunk(message.id, "two ")
    store.append_stream_chunk(message.id, "three")

    assert store.messages_for_session(session.id)[0].content == "one two three"
    assert store._stream_chunks_by_message[message.id] == ["one two three"]

    store.append_stream_chunk(message.id, " four")
    store.append_stream_chunk(message.id, " five")

    assert (
        store.messages_for_session(session.id)[0].content == "one two three four five"
    )
    assert store._stream_chunks_by_message[message.id] == ["one two three four five"]


def test_materialize_between_ticks_is_noop_without_new_chunks():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    store.append_stream_chunk(message.id, "steady")
    first = store.messages_for_session(session.id)[0].content
    second = store.messages_for_session(session.id)[0].content

    assert first == second == "steady"
    assert store._stream_chunks_by_message[message.id] == ["steady"]


def test_read_only_messages_for_session_projects_stream_buffer_without_mutation():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(ephemeral=True)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )
    store.append_stream_chunk(message.id, "seed")
    assert store.messages_for_session(session.id)[0].content == "seed"
    store.append_stream_chunk(message.id, " plus")
    store.append_stream_chunk(message.id, " buffered")

    live = store._message_or_raise(message.id)
    content_before = live.content
    materialized_before = dict(store._stream_materialized_counts)
    payload_before = dict(store._payload_revisions)
    speech_before = dict(store._message_speech_revisions)
    persistence_before = (list(persistence.created), list(persistence.updated))

    projected = store.read_only_messages_for_session(session.id)

    assert projected[0] is not live
    assert projected[0].content == "seed plus buffered"
    assert live.content == content_before == "seed"
    assert store._stream_materialized_counts == materialized_before
    assert store._payload_revisions == payload_before
    assert store._message_speech_revisions == speech_before
    assert (persistence.created, persistence.updated) == persistence_before


def test_collapsed_buffer_keeps_terminal_flush_content_exact():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    store.append_stream_chunk(message.id, "hel")
    # Mid-stream materialize (as the 0.2s tick does) collapses the buffer...
    assert store.messages_for_session(session.id)[0].content == "hel"
    # ...and chunks appended after the collapse still land in the final flush.
    store.append_stream_chunk(message.id, "lo ")
    store.append_stream_chunk(message.id, "world")
    store.mark_message_complete(message.id)

    updated = store.get_message(message.id)
    assert updated.content == "hello world"
    assert updated.status == "complete"


def test_collapsed_buffer_preserves_seeded_retry_content():
    """append_stream_chunk seeds the buffer with existing content; collapsing
    mid-stream must never drop that seed."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    store.append_stream_chunk(message.id, "base")
    assert store.messages_for_session(session.id)[0].content == "base"
    store.mark_message_stopped(message.id)

    # A stopped message keeps partial content; a fresh stream cannot start on
    # it, but the seed path is also exercised by continue-style flows where
    # message.content is non-empty when the first chunk arrives.
    assert store.get_message(message.id).content == "base"


def test_collapsed_buffer_variant_stream_finalizes_full_content():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="original"
    )

    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "re")
    # Tick materialize mid-variant-stream collapses the buffer.
    assert store.messages_for_session(session.id)[0].content == "re"
    store.append_stream_chunk(message.id, "generated")
    store.finalize_variant_stream(message.id)

    updated = store.get_message(message.id)
    assert updated.status == "complete"
    assert updated.content == "regenerated"
    assert [variant.content for variant in updated.variants.variants] == [
        "original",
        "regenerated",
    ]


def test_one_shot_prefill_accessors_round_trip():
    store = ConsoleChatStore()
    session = store.create_session(title="Chat 1")
    assert store.session_one_shot_prefill(session.id) is None
    store.set_session_one_shot_prefill(session.id, "Sure thing:")
    assert store.session_one_shot_prefill(session.id) == "Sure thing:"
    store.set_session_one_shot_prefill(session.id, None)
    assert store.session_one_shot_prefill(session.id) is None


def test_one_shot_prefill_is_per_session():
    store = ConsoleChatStore()
    session_a = store.create_session(title="A")
    session_b = store.create_session(title="B")
    store.set_session_one_shot_prefill(session_a.id, "only A")
    assert store.session_one_shot_prefill(session_b.id) is None


def test_rename_session_persists_conversation_title_when_saved():
    """TASK-341: renaming a saved conversation's tab must rename the
    persisted conversation, not just the ephemeral tab label."""

    class TitleRecordingPersistence(FakePersistence):
        def __init__(self):
            super().__init__()
            self.updated_titles = []

        def update_conversation_title(self, *, conversation_id, title):
            self.updated_titles.append(
                {"conversation_id": conversation_id, "title": title}
            )
            return True

    persistence = TitleRecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.restore_persisted_session(
        title="Old title",
        workspace_id=None,
        persisted_conversation_id="conv-77",
        all_nodes=[],
        active_leaf_persisted_id=None,
    )

    renamed, persisted = store.rename_session(session.id, "New title")

    assert renamed.title == "New title"
    assert persisted is True
    assert persistence.updated_titles == [
        {"conversation_id": "conv-77", "title": "New title"}
    ]


def test_rename_session_keeps_memory_title_when_persistence_fails():
    class ExplodingTitlePersistence(FakePersistence):
        def update_conversation_title(self, *, conversation_id, title):
            raise RuntimeError("db locked")

    store = ConsoleChatStore(persistence=ExplodingTitlePersistence())
    session = store.restore_persisted_session(
        title="Old title",
        workspace_id=None,
        persisted_conversation_id="conv-88",
        all_nodes=[],
        active_leaf_persisted_id=None,
    )

    renamed, persisted = store.rename_session(session.id, "New title")

    assert renamed.title == "New title"
    assert persisted is False


def test_rename_session_without_persisted_conversation_stays_in_memory():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")

    renamed, persisted = store.rename_session(session.id, "Local only")

    assert renamed.title == "Local only"
    assert persisted is True


def test_rename_session_reports_unpersisted_when_update_returns_false():
    """Optimistic-lock/version-check failures surface as persisted=False."""

    class RefusingTitlePersistence(FakePersistence):
        def update_conversation_title(self, *, conversation_id, title):
            return False

    store = ConsoleChatStore(persistence=RefusingTitlePersistence())
    session = store.restore_persisted_session(
        title="Old title",
        workspace_id=None,
        persisted_conversation_id="conv-99",
        all_nodes=[],
        active_leaf_persisted_id=None,
    )

    renamed, persisted = store.rename_session(session.id, "New title")

    assert renamed.title == "New title"
    assert persisted is False


def test_rename_session_reports_unpersisted_when_seam_is_missing():
    """A saved conversation whose persistence lacks the title seam cannot
    have persisted silently — the modal's warning depends on it."""
    # FakePersistence predates update_conversation_title on purpose here.
    store = ConsoleChatStore(persistence=FakePersistence())
    session = store.restore_persisted_session(
        title="Old title",
        workspace_id=None,
        persisted_conversation_id="conv-100",
        all_nodes=[],
        active_leaf_persisted_id=None,
    )

    renamed, persisted = store.rename_session(session.id, "New title")

    assert renamed.title == "New title"
    assert persisted is False


def test_set_session_system_prompt_settings_none_reports_not_applied():
    """task-402: a settings-less session cannot hold the update in memory --
    the method must skip the durable write too and report False, not lie."""
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")
    session.settings = None
    session.persisted_conversation_id = "conv-1"

    updated, persisted = store.set_session_system_prompt(session.id, "New prompt")
    assert persisted is False
    assert updated.settings is None
    assert persistence.updated_system_prompts == []


def test_set_session_pinned_prefill_settings_none_reports_not_applied():
    """task-402: twin contract for the pinned prefill."""
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")
    session.settings = None
    session.persisted_conversation_id = "conv-1"

    updated, persisted = store.set_session_pinned_prefill(session.id, "Voice:")
    assert persisted is False
    assert updated.settings is None
    assert persistence.updated_pinned_prefills == []


# --- task-9: SessionScopeHolder + persist_session_if_needed flush -----------


def test_console_chat_session_gets_its_own_rag_scope_holder():
    """Each session's `rag_scope_holder` starts empty and unshared (mutable
    default-factory sanity check -- a shared instance would leak one
    session's scope into every other session)."""
    first = ConsoleChatSession(title="Chat 1")
    second = ConsoleChatSession(title="Chat 2")

    assert first.rag_scope_holder.scope is None
    assert second.rag_scope_holder.scope is None
    assert first.rag_scope_holder is not second.rag_scope_holder

    first.rag_scope_holder.set(
        RagScope(items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z")
    )
    assert first.rag_scope_holder.scope is not None
    assert second.rag_scope_holder.scope is None


def test_persist_session_if_needed_flushes_held_rag_scope_through_real_db():
    """Drives the REAL `persist_session_if_needed` seam (real in-memory
    `CharactersRAGDB` behind `ChatPersistenceService`, not a hand-rolled
    fake) end to end: a scope held on an unpersisted session's
    `rag_scope_holder` must land in the newly created conversation's
    `metadata["rag_scope"]` at the exact moment first persistence happens."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(title="Chat 1")
        scope = RagScope(
            items=(ScopeItem("media", "m1"), ScopeItem("note", "n1")),
            updated_at="2026-01-01T00:00:00Z",
        )
        session.rag_scope_holder.set(scope)

        conversation_id = store.persist_session_if_needed(session.id)

        assert conversation_id is not None
        assert session.rag_scope_holder.scope is None  # emptied by flush_to
        stored = read_conversation_scope(db, conversation_id)
        assert stored == scope
    finally:
        db.close_connection()


def test_persist_session_if_needed_flushes_rag_scope_exactly_once():
    """A second `persist_session_if_needed` call (the conversation is
    already persisted) must not re-flush -- `flush_to`'s own empties-after-
    flush contract, exercised through the store's real early-return guard."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(title="Chat 1")
        scope = RagScope(
            items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z"
        )
        session.rag_scope_holder.set(scope)

        first_id = store.persist_session_if_needed(session.id)
        second_id = store.persist_session_if_needed(session.id)

        assert first_id == second_id
        assert read_conversation_scope(db, first_id) == scope
        # A later, unrelated holder mutation must never retroactively
        # apply -- the holder is inert after its one-time flush.
        session.rag_scope_holder.set(
            RagScope(
                items=(ScopeItem("media", "m2"),), updated_at="2026-01-02T00:00:00Z"
            )
        )
        store.persist_session_if_needed(session.id)
        assert read_conversation_scope(db, first_id) == scope
    finally:
        db.close_connection()


def test_persist_session_if_needed_without_scope_held_leaves_conversation_unscoped():
    """No scope held -> no `rag_scope` metadata key at all (byte-identical
    to pre-task-9 behavior for the overwhelming common case)."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(title="Chat 1")

        conversation_id = store.persist_session_if_needed(session.id)

        assert read_conversation_scope(db, conversation_id) is None
    finally:
        db.close_connection()


def test_persist_session_if_needed_skips_scope_flush_without_db_seam():
    """A persistence adapter with no `.db` attribute (e.g. the test-only
    `FakePersistence` used throughout this module) must not raise even when
    a scope is held -- the flush is skipped, matching every other durable
    write in this method degrading gracefully when its seam is absent.

    PR #747 review: the loss must also be OBSERVABLE (a warning naming the
    conversation), not merely non-fatal -- silently skipping is exactly how
    a user's pre-persistence scope selection disappears without a trace.
    caplog does not intercept loguru (this project's logger); attach a
    temporary loguru sink instead (mirrors
    ``Tests/Chat/test_attachment_policy.py``'s pattern).
    """
    from loguru import logger as loguru_logger

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")
    session.rag_scope_holder.set(
        RagScope(items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z")
    )

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        conversation_id = store.persist_session_if_needed(session.id)
    finally:
        loguru_logger.remove(sink_id)

    assert conversation_id == "conv-1"
    # The flush was skipped (no seam to write through), so the holder still
    # carries the scope -- nothing was silently lost, it just never landed.
    assert session.rag_scope_holder.scope is not None
    assert any(
        "conv-1" in message and "scope" in message.lower() for message in messages
    ), messages


def test_persist_session_if_needed_no_warning_when_nothing_held_and_no_db_seam():
    """The observability warning is only about LOSS -- a session with no
    scope held must not spuriously warn just because the persistence
    adapter lacks a `.db` seam (nothing was going to be flushed anyway)."""
    from loguru import logger as loguru_logger

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        store.persist_session_if_needed(session.id)
    finally:
        loguru_logger.remove(sink_id)

    assert not any("scope" in message.lower() for message in messages), messages


@pytest.mark.unit
def test_tool_markers_survive_the_next_message():
    """TASK-1842: a follow-up message must not erase the agent's tool trace.

    TOOL markers are deliberately NOT tree nodes -- a marker becoming a
    parent would corrupt the chain for the next real message (see the
    invariant comment in `append_message`). But they were only ever appended
    to `_messages_by_session`, and `_recompute_active_path` is the SINGLE
    writer of that view and rebuilds it from tree nodes alone. So every
    marker was erased by the next ordinary message.

    A user reported tool output appearing then vanishing, replaced by
    `[failed]`. The two are independent: the trace is lost whether or not the
    run fails. Tools are how an agent reaches the outside world, so the
    transcript is the user's only in-context record of what left the machine.
    """
    store = ConsoleChatStore()
    session = store.create_session(title="tool trace")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="a")
    store.append_message(
        session.id, role=ConsoleMessageRole.TOOL, content="⚙ read_file → data"
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.TOOL, content="⚙ search → 3 hits"
    )

    def markers():
        return [
            m.content
            for m in store.messages_for_session(session.id)
            if m.role is ConsoleMessageRole.TOOL
        ]

    assert len(markers()) == 2, "precondition: both markers present during the run"

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="follow-up")
    assert markers() == ["⚙ read_file → data", "⚙ search → 3 hits"], (
        "the follow-up message erased the tool trace"
    )

    # They must sit AFTER the assistant turn they belong to, not float to the
    # end -- otherwise the transcript reads as though the tools ran later.
    contents = [m.content for m in store.messages_for_session(session.id)]
    assert contents.index("⚙ read_file → data") > contents.index("a")
    assert contents.index("⚙ read_file → data") < contents.index("follow-up")

    # And the invariant they exist to protect must still hold: a marker must
    # never become a tree node or the active leaf.
    assert store._active_leaf_by_session[session.id] is not None
    leaf = store._nodes_by_session[session.id][
        store._active_leaf_by_session[session.id]
    ]
    assert leaf.role is not ConsoleMessageRole.TOOL


@pytest.mark.unit
def test_closing_a_session_releases_its_tool_markers():
    """TASK-1842 follow-up: `_tool_markers_by_session` outlived the session.

    `close_session` pops every other per-session structure and sweeps owned
    ids out of `_message_session_index`, but left the marker registry keyed
    by a dead session id -- so every TOOL marker object a closed session ever
    produced was retained for the life of the process.
    """
    store = ConsoleChatStore()
    session = store.create_session(title="tool trace")
    other = store.create_session(title="keep me")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="a")
    store.append_message(
        session.id, role=ConsoleMessageRole.TOOL, content="⚙ read_file → data"
    )
    assert store._tool_markers_by_session.get(session.id), "precondition"

    store.close_session(session.id)

    assert session.id not in store._tool_markers_by_session, (
        "the closed session's markers are still retained"
    )
    assert other.id in store._sessions, "closing one session must not touch others"


@pytest.mark.unit
def test_deleting_an_anchor_node_purges_the_markers_it_anchored():
    """TASK-1842 follow-up: deleted branches left dangling marker bookkeeping.

    `delete_message` purges the whole subtree from every node structure and
    from `_message_session_index`, but it could not reach display-only marker
    ids -- markers are not tree nodes. Their anchor was gone, so they never
    rendered again (`_with_tool_markers` drops off-path anchors), yet both the
    marker objects and their `_message_session_index` entries survived,
    claiming a session still owned messages it could never show.
    """
    store = ConsoleChatStore()
    session = store.create_session(title="tool trace")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q")
    answer = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="a"
    )
    marker = store.append_message(
        session.id, role=ConsoleMessageRole.TOOL, content="⚙ read_file → data"
    )
    assert store._message_session_index.get(marker.id) == session.id, "precondition"

    store.delete_message(answer.id)

    assert marker.id not in store._message_session_index, (
        "the marker's index entry outlived the node it was anchored to"
    )
    assert not any(
        anchor == answer.id
        for anchor, _marker in store._tool_markers_by_session.get(session.id, [])
    ), "a marker is still anchored to a deleted node"


def test_set_message_usage_on_a_streaming_message_defers_persistence():
    """The normal ordering: usage lands on a still-streaming message and the
    TERMINAL mark that follows is what flushes it (one write, not two)."""
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "hi")
    usage = ProviderUsage(
        uncached_input=10, output=5, provider="openai", model="gpt-4o"
    )

    updated = store.set_message_usage(message.id, usage)

    assert updated.usage == usage
    assert store.get_message(message.id).usage == usage
    assert store.get_message(message.id).status == "streaming"
    assert persistence.updated == [], "a streaming message must not flush early"


def test_set_message_usage_after_a_terminal_mark_flushes_to_persistence():
    """Final-review F3: on the Stop path the message is finalized BEFORE the
    cancelled task attaches its partial usage, so the terminal mark cannot
    flush it -- the attach itself has to. Without this, a stopped turn's
    already-billed input tokens never reached the DB.
    """
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    class UsageUpdatePersistence(RecordingPersistence):
        def __init__(self):
            super().__init__()
            self.usage_values = []

        def update_message_content(self, *, usage_json=None, **kwargs):
            self.usage_values.append(usage_json)
            return super().update_message_content(**kwargs)

    persistence = UsageUpdatePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "partial answer")
    stopped = store.mark_message_stopped(message.id)
    assert stopped.status == "stopped"
    assert all(value is None for value in persistence.usage_values)

    store.set_message_usage(
        message.id,
        ProviderUsage(
            uncached_input=3571,
            cache_read=6656,
            provider="anthropic",
            model="claude-sonnet-5",
            partial=True,
        ),
    )

    assert persistence.usage_values[-1] is not None
    assert '"uncached_input": 3571' in persistence.usage_values[-1]
    assert '"cache_read": 6656' in persistence.usage_values[-1]
    assert '"partial": true' in persistence.usage_values[-1]


def test_stop_path_usage_flush_uses_local_write_and_leaves_version_unchanged():
    """Qodo round (Finding 4), AC (d)(ii): the same Stop-path late-usage-
    attach flush as the F3 test above, but against a REAL
    ``ChatPersistenceService``/``CharactersRAGDB`` pair instead of a hand-
    rolled fake -- proving the usage-only flush actually lands through
    ``update_message_usage_local`` (no version/last_modified bump, no
    ``sync_log`` row) rather than only through a fake that can't observe
    that distinction.
    """
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.ensure_session(title="Chat 1")
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        store.append_stream_chunk(message.id, "partial answer")
        stopped = store.mark_message_stopped(message.id)
        assert stopped.status == "stopped"

        persisted_id = store.get_message(message.id).persisted_message_id
        assert persisted_id is not None
        row_after_stop = db.get_message_by_id(persisted_id)
        assert row_after_stop["usage_json"] is None
        version_after_stop = row_after_stop["version"]
        last_modified_after_stop = row_after_stop["last_modified"]
        change_id_after_stop = db.get_latest_sync_log_change_id()

        store.set_message_usage(
            message.id,
            ProviderUsage(
                uncached_input=3571,
                cache_read=6656,
                provider="anthropic",
                model="claude-sonnet-5",
                partial=True,
            ),
        )

        row_after_usage = db.get_message_by_id(persisted_id)
        assert row_after_usage["usage_json"] is not None
        assert '"uncached_input": 3571' in row_after_usage["usage_json"]
        assert '"cache_read": 6656' in row_after_usage["usage_json"]
        # The load-bearing assertion: the usage-only flush did NOT bump
        # version/last_modified a second time on top of the stop flush.
        assert row_after_usage["version"] == version_after_stop
        assert row_after_usage["last_modified"] == last_modified_after_stop

        new_entries = db.get_sync_log_entries(
            since_change_id=change_id_after_stop, entity_type="messages"
        )
        assert new_entries == [], (
            "the usage-only local flush must not enqueue a sync_log row"
        )
    finally:
        db.close_connection()


class _UsageUpdatePersistence(RecordingPersistence):
    """RecordingPersistence that keeps every usage_json it is handed."""

    def __init__(self):
        super().__init__()
        self.usage_values = []

    def update_message_content(self, *, usage_json=None, **kwargs):
        self.usage_values.append(usage_json)
        return super().update_message_content(**kwargs)

    def last_usage(self):
        recorded = [value for value in self.usage_values if value is not None]
        return recorded[-1] if recorded else None


def _completed_message_with_usage(store, session, usage):
    """Stream an answer to completion with `usage` recorded against it."""
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "the original answer")
    store.set_message_usage(message.id, usage)
    store.mark_message_complete(message.id)
    return message


@pytest.mark.parametrize("attach_before_mark", [False, True])
def test_stopped_regenerate_keeps_the_original_answers_usage(attach_before_mark):
    """A stopped regenerate must not price the ORIGINAL answer with the
    abandoned run's numbers.

    ``mark_message_stopped`` restores a mid-regenerate message to its
    pre-regenerate content AND status, so the message ends up "complete"
    again, showing the original answer. The abandoned run's cancelled task
    then attaches its partial usage -- and the terminal flush added for the
    Stop path (F3) wrote it straight over the original's durable record.

    Both real orderings are pinned:
      * ``attach_before_mark=False`` -- ``stop_active_run`` finalizes the
        message first, then cancels the task whose handler attaches (the
        empirically reproduced case).
      * ``attach_before_mark=True`` -- the in-loop cancel check attaches
        before calling ``_mark_stream_stopped``.
    """
    persistence = _UsageUpdatePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    original_usage = ProviderUsage(
        uncached_input=1200,
        output=340,
        provider="anthropic",
        model="claude-sonnet-5",
        partial=False,
    )
    message = _completed_message_with_usage(store, session, original_usage)
    assert store.get_message(message.id).usage == original_usage
    assert '"uncached_input": 1200' in persistence.last_usage()

    # Regenerate: the pre-regenerate content, status AND usage are snapshotted.
    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "half of a new ans")

    abandoned_usage = ProviderUsage(
        output=7, provider="anthropic", model="claude-sonnet-5", partial=True
    )
    if attach_before_mark:
        store.set_message_usage(message.id, abandoned_usage)
        stopped = store.mark_message_stopped(message.id)
    else:
        stopped = store.mark_message_stopped(message.id)
        store.set_message_usage(message.id, abandoned_usage)

    # Restored to the original generation in every respect.
    assert stopped.content == "the original answer"
    assert stopped.status == "complete"
    current = store.get_message(message.id)
    assert current.content == "the original answer"
    assert current.usage == original_usage
    assert current.usage.partial is False

    persisted = persistence.last_usage()
    assert '"uncached_input": 1200' in persisted
    assert '"output": 340' in persisted
    assert '"partial": false' in persisted
    assert '"output": 7' not in persisted


def test_regenerating_again_after_a_stopped_regenerate_records_usage_normally():
    """The guard is scoped to the abandoned run, not to the message: a fresh
    regenerate re-arms usage capture."""
    persistence = _UsageUpdatePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    message = _completed_message_with_usage(
        store,
        session,
        ProviderUsage(uncached_input=1200, output=340, provider="anthropic", model="m"),
    )
    store.begin_variant_stream(message.id)
    store.mark_message_stopped(message.id)
    store.set_message_usage(
        message.id,
        ProviderUsage(output=7, provider="anthropic", model="m", partial=True),
    )

    # Second regenerate, this one succeeds.
    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "a better answer")
    second_usage = ProviderUsage(
        uncached_input=1500, output=400, provider="anthropic", model="m"
    )
    store.set_message_usage(message.id, second_usage)
    store.finalize_variant_stream(message.id)

    assert store.get_message(message.id).usage == second_usage
    assert '"uncached_input": 1500' in persistence.last_usage()


def test_failed_regenerate_keeps_the_original_answers_usage():
    """``mark_message_failed`` restores the same pre-regenerate state as
    ``mark_message_stopped``; the agent path attaches ahead of BOTH terminal
    marks, so the same clobber applies."""
    persistence = _UsageUpdatePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    original_usage = ProviderUsage(
        uncached_input=1200, output=340, provider="anthropic", model="m"
    )
    message = _completed_message_with_usage(store, session, original_usage)

    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "half a")
    store.mark_message_failed(message.id)
    store.set_message_usage(
        message.id,
        ProviderUsage(output=7, provider="anthropic", model="m", partial=True),
    )

    assert store.get_message(message.id).usage == original_usage
    assert '"output": 7' not in persistence.last_usage()


def test_set_message_usage_unknown_id_raises_keyerror():
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    store = ConsoleChatStore()
    store.ensure_session(title="Chat 1")
    with pytest.raises(KeyError):
        store.set_message_usage("missing", ProviderUsage())


def test_terminal_flush_passes_usage_json_to_accepting_persistence():
    """``mark_message_complete`` first materializes the streamed content
    (a create through ``_persist_pending_message_if_ready``), then flushes
    the terminal status through ``_persist_existing_message`` -- which now
    has a ``persisted_message_id`` and so calls ``update_message_content``.
    Usage set before completion must ride that final update call."""
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    class UsagePersistence(RecordingPersistence):  # RecordingPersistence at :1792
        def __init__(self):
            super().__init__()
            self.update_usage_values = []

        def update_message_content(self, *, usage_json=None, **kwargs):
            self.update_usage_values.append(usage_json)
            return super().update_message_content(**kwargs)

    persistence = UsagePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "hello")
    store.set_message_usage(
        message.id,
        ProviderUsage(uncached_input=10, output=2, provider="openai", model="gpt-4o"),
    )

    store.mark_message_complete(message.id)

    assert persistence.update_usage_values
    stored = persistence.update_usage_values[-1]
    assert stored is not None and '"uncached_input": 10' in stored


def test_narrow_persistence_without_usage_kwarg_still_works():
    # FakePersistence (:573) declares keyword-only params and no
    # **kwargs -- the _persistence_accepts_kwarg probe must skip usage_json.
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "hello")
    store.set_message_usage(message.id, ProviderUsage(uncached_input=1))

    completed = store.mark_message_complete(message.id)  # must not raise
    assert completed.status == "complete"


def test_payload_revision_bumps_on_payload_mutations():
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    r0 = store.payload_revision(session.id)

    message = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hi"
    )
    r1 = store.payload_revision(session.id)
    assert r1 > r0

    store.update_message_content(message.id, "edited")
    r2 = store.payload_revision(session.id)
    assert r2 > r1

    reply = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="yo"
    )
    r3 = store.payload_revision(session.id)
    store.set_message_usage(
        reply.id, ProviderUsage(uncached_input=1, provider="anthropic", model="m")
    )
    # usage attach is NOT payload-affecting
    assert store.payload_revision(session.id) == r3


def test_payload_revision_bumps_on_settings_and_system_prompt():
    from dataclasses import replace

    store = ConsoleChatStore()
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )
    r0 = store.payload_revision(session.id)
    store.set_session_system_prompt(session.id, "be terse")
    r1 = store.payload_revision(session.id)
    assert r1 > r0
    store.replace_session_settings(
        session.id, replace(session.settings, model="claude-sonnet-4-6")
    )
    assert store.payload_revision(session.id) > r1


def test_payload_revision_not_bumped_per_stream_chunk():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    r0 = store.payload_revision(session.id)
    store.append_stream_chunk(message.id, "a")
    store.append_stream_chunk(message.id, "b")
    assert store.payload_revision(session.id) == r0  # chunks don't churn
    store.mark_message_complete(message.id)
    assert store.payload_revision(session.id) > r0  # completion does


def test_append_message_metadata_rides_the_create_write():
    """task-2364: structured metadata is written with the row, not by a
    follow-up update -- a realtime row knows its engine/provider/model at
    creation."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    metadata = MessageMetadata(
        engine="realtime", provider="openai", model="gpt-realtime"
    )

    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="spoken answer",
        persist=True,
        metadata=metadata,
    )

    assert message.metadata == metadata
    assert persistence.created[-1]["metadata_json"] == metadata.to_json()


def test_default_metadata_is_never_written():
    """An all-default instance carries no facts; writing it would store a
    row of noise indistinguishable from "nothing known"."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="typed, not spoken",
        persist=True,
        metadata=MessageMetadata(),
    )

    assert "metadata_json" not in persistence.created[-1]


def test_set_message_metadata_on_an_unpersisted_row_rides_the_later_create():
    """A realtime user row is created EMPTY (deferred persistence) and only
    reaches the DB once its transcript lands -- the status recorded in
    between must travel with that create."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="",
        persist=True,
        metadata=MessageMetadata(engine="realtime", transcript_status="pending"),
    )
    assert persistence.created == [], "an empty row has nothing to persist yet"

    store.set_message_metadata(
        message.id,
        MessageMetadata(engine="realtime", transcript_status="final"),
    )
    store.finalize_deferred_user_message_content(message.id, "what the user said")

    assert persistence.created[-1]["content"] == "what the user said"
    assert '"transcript_status": "final"' in persistence.created[-1]["metadata_json"]


def test_finalize_deferred_user_message_content_preserves_reply_descendant():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="",
        persist=True,
    )
    reply = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )

    store.finalize_deferred_user_message_content(user.id, "what the user said")

    rows = store.messages_for_session(session.id)
    assert [row.id for row in rows] == [user.id, reply.id]
    assert rows[0].content == "what the user said"
    assert store.get_message(reply.id).role is ConsoleMessageRole.ASSISTANT


def test_an_empty_transcript_placeholder_persists_through_the_deferred_create():
    """task-2391: a committed voice turn whose transcript comes back with no
    words must still survive a restart. The store defers persistence for a
    content-less row (same guard proven above), and the DB layer refuses to
    create a message with neither text nor an image at all
    (`CharactersRAGDB.add_message`) -- so a metadata-only "empty" record can
    never durably exist. Writing a short, honest placeholder as the row's
    real content -- through the same `finalize_deferred_user_message_content`
    call used by the "final" transcript case above -- flushes the deferred
    create, and a follow-up metadata-only patch (mirroring the "final" case's own
    two-step order: content write, then status write) marks it "empty"."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
    )

    placeholder = CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="",
        persist=True,
        metadata=MessageMetadata(engine="realtime", transcript_status="pending"),
    )
    assert persistence.created == [], "an empty row has nothing to persist yet"

    store.finalize_deferred_user_message_content(message.id, placeholder)
    assert persistence.created[-1]["content"] == placeholder, (
        "the row must be durably created once its emptiness is final, not "
        "left stranded in memory"
    )

    store.set_message_metadata(
        message.id,
        MessageMetadata(engine="realtime", transcript_status="empty"),
    )
    assert '"transcript_status": "empty"' in persistence.updated[-1]["metadata_json"]


def test_empty_transcript_placeholder_reaches_a_real_db_through_the_deferred_create():
    """Qodo Q2 (task-2391 review): the sibling test above proves WHICH
    persistence calls happen and with what kwargs via `RecordingPersistence`
    -- this file's established seam for that class of claim (see also its
    own direct precedent, `test_set_message_metadata_on_an_unpersisted_row_
    rides_the_later_create`, and the file-wide count: 9 `RecordingPersistence`
    call-shape tests vs. 5 real-DB tests, each of the latter carrying its own
    docstring explaining why a fake can't stand in there). A fake cannot
    observe whether a REAL DB actually accepts and durably stores the row,
    though, and `Tests/UI/test_console_resume_active_path.py::test_resume_
    restores_an_empty_transcript_row_and_its_explanation` -- the existing
    real-DB coverage for this exact row shape -- HAND-SEEDS the row directly
    via `db.add_message(...)`; it never exercises
    `finalize_deferred_user_message_content`'s deferred-create flush at all,
    so it does not close this gap either. This test does: drives the real flow
    (deferred row -> content write -> status write) against a real
    `ChatPersistenceService`/`CharactersRAGDB` pair and reads the row straight
    back off the DB, mirroring this file's own
    established real-DB pattern (`test_persist_session_if_needed_flushes_
    held_rag_scope_through_real_db`, `test_stop_path_usage_flush_uses_local_
    write_and_leaves_version_unchanged`) for exactly this kind of durability
    claim."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
    )

    db = CharactersRAGDB(":memory:", "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.ensure_session(title="Chat 1")
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="",
            persist=True,
            metadata=MessageMetadata(engine="realtime", transcript_status="pending"),
        )
        assert message.persisted_message_id is None, "deferred, not yet durable"

        store.finalize_deferred_user_message_content(
            message.id, CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
        )
        store.set_message_metadata(
            message.id,
            MessageMetadata(engine="realtime", transcript_status="empty"),
        )

        persisted_id = store.get_message(message.id).persisted_message_id
        assert persisted_id is not None, "the deferred create must have flushed"
        row = db.get_message_by_id(persisted_id)
        assert row["content"] == CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
        assert '"transcript_status": "empty"' in row["metadata_json"]
    finally:
        db.close_connection()


def test_set_message_metadata_flushes_locally_and_leaves_the_version_alone():
    """Same local-only contract as the usage flush, against a REAL
    persistence/DB pair: metadata is this device's own observation, so the
    write must not bump version/last_modified or enqueue a sync_log row.
    """
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    db = CharactersRAGDB(":memory:", "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.ensure_session(title="Chat 1")
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        store.append_stream_chunk(message.id, "half a sen")
        store.mark_message_complete(message.id)

        persisted_id = store.get_message(message.id).persisted_message_id
        assert persisted_id is not None
        row_before = db.get_message_by_id(persisted_id)
        assert row_before["metadata_json"] is None
        change_id = db.get_latest_sync_log_change_id()

        store.set_message_metadata(
            message.id,
            MessageMetadata(
                engine="realtime",
                provider="openai",
                model="gpt-realtime",
                interrupted=True,
            ),
        )

        row_after = db.get_message_by_id(persisted_id)
        assert '"interrupted": true' in row_after["metadata_json"]
        assert row_after["version"] == row_before["version"]
        assert row_after["last_modified"] == row_before["last_modified"]
        assert (
            db.get_sync_log_entries(since_change_id=change_id, entity_type="messages")
            == []
        )
    finally:
        db.close_connection()


def _seeded_roleplay_store():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(
        title="Chat with Alraune",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
        character_name="Alraune",
    )
    greeting = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Hello User.",
        persist=True,
        metadata=MessageMetadata(
            template_kind="character_greeting",
            template_source="Hello {{user}}.",
        ),
    )
    session.character_system_template = "Speak with {{user}}."
    session.settings = ConsoleSessionSettings(
        provider="llama_cpp", system_prompt="Speak with User."
    )
    return store, persistence, session, greeting


def test_session_override_is_not_console_session_settings():
    session = ConsoleChatSession(user_display_name_override="Rowan")

    assert session.user_display_name_override == "Rowan"
    assert not hasattr(
        ConsoleSessionSettings(provider="llama_cpp"), "user_display_name_override"
    )


def test_first_persist_flushes_roleplay_context_after_conversation_exists():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp"),
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
        character_name="Alraune",
    )
    session.user_display_name_override = "Rowan"
    session.character_system_template = "Speak with {{user}}."

    conversation_id = store.persist_session_if_needed(session.id)

    assert persistence.roleplay_updates == [
        {
            "conversation_id": conversation_id,
            "user_name_override": "Rowan",
            "character_system_template": "Speak with {{user}}.",
            "character_name_snapshot": "Alraune",
        }
    ]


def test_temporary_session_keeps_override_without_durable_write():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(ephemeral=True)

    updated, persisted = store.set_session_user_display_name_override(
        session.id, "Rowan", global_default="User"
    )

    assert updated.user_display_name_override == "Rowan"
    assert persisted is True
    assert persistence.roleplay_updates == []


def test_rename_rematerializes_system_and_seeded_greeting():
    store, persistence, session, greeting = _seeded_roleplay_store()

    _updated, persisted = store.set_session_user_display_name_override(
        session.id, "Captain Rowan", global_default="User"
    )

    assert persisted is True
    assert session.settings.system_prompt == "Speak with Captain Rowan."
    assert store.get_message(greeting.id).content == "Hello Captain Rowan."
    assert persistence.updated_messages[-1]["content"] == "Hello Captain Rowan."


def test_editing_derived_greeting_clears_template_provenance():
    store, _persistence, _session, greeting = _seeded_roleplay_store()

    edited = store.update_message_content(greeting.id, "Hello there.")

    assert edited.metadata is not None
    assert edited.metadata.template_kind == ""
    assert edited.metadata.template_source == ""


def test_editing_system_prompt_clears_character_template_source():
    store, persistence, session, _greeting = _seeded_roleplay_store()

    updated, persisted = store.set_session_system_prompt(session.id, "Be concise.")

    assert persisted is True
    assert updated.character_system_template is None
    assert persistence.roleplay_updates[-1]["character_system_template"] is None


def test_refresh_roleplay_projections_is_idempotent_when_values_are_current():
    store, persistence, session, _greeting = _seeded_roleplay_store()
    store.set_session_user_display_name_override(
        session.id, "Rowan", global_default="User"
    )
    revision = store.payload_revision(session.id)
    update_count = len(persistence.updated_messages)

    persisted = store.refresh_session_roleplay_projections(
        session.id, global_default="User"
    )

    assert persisted is True
    assert store.payload_revision(session.id) == revision
    assert len(persistence.updated_messages) == update_count


def test_editing_derived_greeting_persists_cleared_metadata():
    store, persistence, _session, greeting = _seeded_roleplay_store()

    store.update_message_content(greeting.id, "Hello there.")

    assert (
        persistence.updated_messages[-1]["metadata_json"] == MessageMetadata().to_json()
    )


def test_falsy_projection_write_reports_unpersisted_without_sync():
    class RefusingPersistence(FakePersistence):
        def update_message_content(self, **kwargs):
            super().update_message_content(**kwargs)
            return False

    persistence = RefusingPersistence()
    store, _unused, session, _greeting = _seeded_roleplay_store()
    store.persistence = persistence

    _updated, persisted = store.set_session_user_display_name_override(
        session.id, "Rowan", global_default="User"
    )

    assert persisted is False


def test_falsy_system_prompt_write_reports_unpersisted():
    class RefusingPersistence(FakePersistence):
        def update_conversation_system_prompt(self, **kwargs):
            return False

    persistence = RefusingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp")
    )
    store.persist_session_if_needed(session.id)

    _updated, persisted = store.set_session_system_prompt(session.id, "Be concise.")

    assert persisted is False


def test_successful_regeneration_clears_greeting_provenance():
    store, persistence, _session, greeting = _seeded_roleplay_store()

    store.begin_variant_stream(greeting.id)
    store.append_stream_chunk(greeting.id, "A generated reply.")
    completed = store.finalize_variant_stream(greeting.id)

    assert completed.metadata == MessageMetadata()
    assert (
        persistence.updated_messages[-1]["metadata_json"] == MessageMetadata().to_json()
    )


def test_stopped_regeneration_restores_greeting_provenance():
    store, _persistence, _session, greeting = _seeded_roleplay_store()

    store.begin_variant_stream(greeting.id)
    restored = store.mark_message_stopped(greeting.id)

    assert restored.metadata is not None
    assert restored.metadata.template_kind == "character_greeting"


def test_identity_revisions_track_provenance_and_character_name_changes():
    store, _persistence, session, greeting = _seeded_roleplay_store()
    identity_before = session.identity_revision
    payload_before = store.payload_revision(session.id)

    store.update_message_content(greeting.id, "Manual greeting.")

    assert session.identity_revision == identity_before + 1
    assert store.payload_revision(session.id) == payload_before + 1
    store.set_session_character_name(session.id, "Nyx", global_default="User")
    assert session.identity_revision == identity_before + 2
    assert store.payload_revision(session.id) == payload_before + 2


def test_character_name_and_seed_are_idempotent_when_unchanged():
    store, _persistence, session, _greeting = _seeded_roleplay_store()
    revision = session.identity_revision
    payload = store.payload_revision(session.id)

    store.set_session_character_name(session.id, "Alraune", global_default="User")
    store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="",
        global_default="User",
    )

    assert session.identity_revision == revision
    assert store.payload_revision(session.id) == payload
    assert (
        store.presentation_context(session.id, "Captain Rowan").character_name
        == "Alraune"
    )


def test_character_roleplay_swap_persists_only_the_final_projection_and_context():
    """Separate name/template mutations expose a hybrid durable projection."""
    store, persistence, session, _greeting = _seeded_roleplay_store()
    persistence.updated_system_prompts.clear()
    persistence.roleplay_updates.clear()

    updated, greeting, persisted = store.swap_session_character_roleplay(
        session.id,
        character_name="Brynn",
        system_template="Serve {{user}} as {{character}}.",
        greeting_template="",
        global_default="Captain Rowan",
    )

    assert persisted is True
    assert greeting is None
    assert updated.character_name == "Brynn"
    assert updated.character_system_template == "Serve {{user}} as {{character}}."
    assert updated.settings.system_prompt == "Serve Captain Rowan as Brynn."
    assert persistence.updated_system_prompts == [
        {
            "conversation_id": "conv-1",
            "system_prompt": "Serve Captain Rowan as Brynn.",
        }
    ]
    assert persistence.roleplay_updates == [
        {
            "conversation_id": "conv-1",
            "user_name_override": None,
            "character_system_template": "Serve {{user}} as {{character}}.",
            "character_name_snapshot": "Brynn",
        }
    ]


def test_first_persist_context_failure_does_not_force_atomic_promotion_legacy_path():
    class RefusingPersistence(FakePersistence):
        def update_conversation_roleplay_context(self, **kwargs):
            return False

    persistence = RefusingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    saved = store.create_session()
    saved.user_display_name_override = "Rowan"
    assert store.persist_session_if_needed(saved.id) == "conv-1"
    assert saved.persisted_conversation_id == "conv-1"

    temporary = store.create_session(
        ephemeral=True,
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
        character_name="Alraune",
    )
    temporary.user_display_name_override = "Rowan"
    conversation_id = store.promote_ephemeral_session(temporary.id)

    assert conversation_id is not None
    assert temporary.ephemeral is False
    assert temporary.persisted_conversation_id == conversation_id
    roleplay = persistence.last_create_kwargs["metadata"]["console_roleplay_context"]
    assert roleplay["user_name_override"] == "Rowan"
    assert roleplay["character_name_snapshot"] == "Alraune"


def test_generic_roleplay_context_does_not_capture_a_character_name():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(settings=ConsoleSessionSettings(provider="llama_cpp"))
    session.user_display_name_override = "Rowan"

    conversation_id = store.persist_session_if_needed(session.id)

    assert persistence.roleplay_updates == [
        {
            "conversation_id": conversation_id,
            "user_name_override": "Rowan",
            "character_system_template": None,
            "character_name_snapshot": None,
        }
    ]


def test_identical_real_seed_does_not_append_a_duplicate_greeting():
    store, _persistence, session, _greeting = _seeded_roleplay_store()
    message_count = len(store.messages_for_session(session.id))
    identity = session.identity_revision
    payload = store.payload_revision(session.id)

    duplicate = store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="User",
    )

    assert duplicate is None
    assert len(store.messages_for_session(session.id)) == message_count
    assert session.identity_revision == identity
    assert store.payload_revision(session.id) == payload


def test_changed_seed_source_can_append_a_new_greeting():
    store, _persistence, session, _greeting = _seeded_roleplay_store()
    message_count = len(store.messages_for_session(session.id))

    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Respond warmly to {{user}}.",
        greeting_template="Welcome {{user}}.",
        global_default="User",
    )

    assert greeting is not None
    assert len(store.messages_for_session(session.id)) == message_count + 1


def test_falsy_projection_write_does_not_enqueue_sync():
    class RefusingPersistence(FakePersistence):
        def update_message_content(self, **kwargs):
            super().update_message_content(**kwargs)
            return False

    store, _unused, session, _greeting = _seeded_roleplay_store()
    sync = FakeChatSyncProducer()
    store.persistence = RefusingPersistence()
    store.sync_v2_chat_producer = sync
    store.sync_v2_server_profile_id = "profile-1"

    _updated, persisted = store.set_session_user_display_name_override(
        session.id, "Rowan", global_default="User"
    )

    assert persisted is False
    assert sync.enqueued == []


def test_durable_clear_replaces_previously_persisted_greeting_provenance():
    store, persistence, _session, greeting = _seeded_roleplay_store()
    original = persistence.created_messages[-1]["metadata_json"]
    assert original == greeting.metadata.to_json()

    store.update_message_content(greeting.id, "Manual greeting.")

    assert (
        persistence.updated_messages[-1]["metadata_json"] == MessageMetadata().to_json()
    )


def test_stale_refresh_rematerializes_and_reports_success_once():
    store, persistence, session, greeting = _seeded_roleplay_store()

    assert (
        store.refresh_session_roleplay_projections(session.id, global_default="Rowan")
        is True
    )
    assert store.get_message(greeting.id).content == "Hello Rowan."
    writes = len(persistence.updated_messages)
    revision = store.payload_revision(session.id)
    assert (
        store.refresh_session_roleplay_projections(session.id, global_default="Rowan")
        is True
    )
    assert len(persistence.updated_messages) == writes
    assert store.payload_revision(session.id) == revision


def test_prepare_roleplay_refresh_materializes_live_before_immutable_persistence():
    store, persistence, session, greeting = _seeded_roleplay_store()
    persistence.updated_messages.clear()
    persistence.updated_system_prompts.clear()

    plan = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Captain Rowan"
    )

    assert plan is not None
    assert session.settings.system_prompt == "Speak with Captain Rowan."
    assert store.get_message(greeting.id).content == "Hello Captain Rowan."
    assert persistence.updated_system_prompts == []
    assert persistence.updated_messages == []
    with pytest.raises(FrozenInstanceError):
        plan.generation = -1
    assert plan.system_prompt_write is not None
    assert plan.system_prompt_write.expected_roleplay_context == ConsoleRoleplayContext(
        character_system_template="Speak with {{user}}.",
        character_name_snapshot="Alraune",
    )

    store.close_session(session.id)
    result = ConsoleChatStore.persist_roleplay_projection_plan(plan)

    assert result.persisted is True
    assert persistence.updated_system_prompts[-1]["system_prompt"] == (
        "Speak with Captain Rowan."
    )
    assert persistence.updated_messages[-1]["content"] == "Hello Captain Rowan."
    assert store.accept_roleplay_projection_persistence_result(result) is False


def test_forced_roleplay_repair_snapshots_current_sources_without_revision_bumps():
    store, persistence, session, greeting = _seeded_roleplay_store()
    identity_revision = session.identity_revision
    payload_revision = store.payload_revision(session.id)
    speech_revision = store._message_speech_revisions.get(greeting.id)
    persistence.updated_system_prompts.clear()
    persistence.updated_messages.clear()

    plan = store.prepare_session_roleplay_projection_refresh(
        session.id,
        global_default="User",
        force_persistence=True,
    )

    assert plan is not None
    assert plan.system_prompt_write is not None
    assert len(plan.message_writes) == 1
    assert session.identity_revision == identity_revision
    assert store.payload_revision(session.id) == payload_revision
    assert store._message_speech_revisions.get(greeting.id) == speech_revision
    result = store.persist_roleplay_projection_plan(plan)
    assert store.accept_roleplay_projection_persistence_result(result) is True
    assert persistence.updated_system_prompts[-1]["system_prompt"] == (
        "Speak with User."
    )
    assert persistence.updated_messages[-1]["content"] == "Hello User."


def test_forced_restored_roleplay_repair_accepts_owned_alpha_ancestor(tmp_path):
    db = CharactersRAGDB(tmp_path / "restored-roleplay-repair.db", "task-5")
    try:
        service = ChatPersistenceService(db)
        conversation_id = service.create_conversation(
            assistant_kind="generic",
            assistant_id="console",
            system_prompt="Speak with Alpha.",
        )
        assert (
            service.update_conversation_roleplay_context(
                conversation_id=conversation_id,
                user_name_override=None,
                character_system_template="Speak with {{user}}.",
                character_name_snapshot="Alraune",
            )
            is True
        )
        greeting_metadata = MessageMetadata(
            template_kind="character_greeting",
            template_source="Hello {{user}}.",
        )
        persisted_message_id = service.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            content="Hello Alpha.",
            metadata_json=greeting_metadata.to_json(),
        )
        stale_store = ConsoleChatStore(persistence=service)
        stale_session = stale_store.create_session(
            settings=ConsoleSessionSettings(
                provider="llama_cpp", system_prompt="Speak with Bravo."
            ),
            assistant_kind="character",
            character_name="Alraune",
        )
        stale_session.persisted_conversation_id = conversation_id
        stale_session.character_system_template = "Speak with {{user}}."
        stale_greeting = stale_store.append_message(
            stale_session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Hello Bravo.",
            persist=False,
            metadata=greeting_metadata,
        )
        stale_store._nodes_by_session[stale_session.id][
            stale_greeting.id
        ].persisted_message_id = persisted_message_id
        stale_plan = stale_store.prepare_session_roleplay_projection_refresh(
            stale_session.id,
            global_default="Bravo",
            force_persistence=True,
        )
        assert stale_plan is not None
        assert stale_plan.system_prompt_write is not None
        assert stale_plan.system_prompt_write.source_owned_repair is True
        assert stale_plan.message_writes[0].source_owned_repair is True

        store = ConsoleChatStore(persistence=service)
        session = store.create_session(
            settings=ConsoleSessionSettings(
                provider="llama_cpp", system_prompt="Speak with Cecelia."
            ),
            assistant_kind="character",
            character_name="Alraune",
        )
        session.persisted_conversation_id = conversation_id
        session.character_system_template = "Speak with {{user}}."
        greeting = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Hello Cecelia.",
            persist=False,
            metadata=greeting_metadata,
        )
        store._nodes_by_session[session.id][
            greeting.id
        ].persisted_message_id = persisted_message_id

        plan = store.prepare_session_roleplay_projection_refresh(
            session.id,
            global_default="Cecelia",
            force_persistence=True,
        )
        assert plan is not None
        assert len(plan.message_writes) == 1
        result = store.persist_roleplay_projection_plan(plan)

        assert result.persisted is True
        assert store.accept_roleplay_projection_persistence_result(result) is True
        assert db.get_conversation_by_id(conversation_id)["system_prompt"] == (
            "Speak with Cecelia."
        )
        assert db.get_message_by_id(persisted_message_id)["content"] == (
            "Hello Cecelia."
        )
        stale_result = stale_store.persist_roleplay_projection_plan(stale_plan)
        assert stale_result.persisted is False
        assert db.get_conversation_by_id(conversation_id)["system_prompt"] == (
            "Speak with Cecelia."
        )
        assert db.get_message_by_id(persisted_message_id)["content"] == (
            "Hello Cecelia."
        )
    finally:
        db.close_connection()


@pytest.mark.parametrize("execute_b", (True, False), ids=("b-applied", "b-skipped"))
def test_accepted_roleplay_sync_rebases_c_from_latest_owned_outbox_hash(
    tmp_path, execute_b
):
    class ChatTarget:
        def __init__(self) -> None:
            self.hashes: dict[str, str] = {}
            self.messages: dict[str, dict] = {}
            self.conflicts: list[dict] = []

        def get_chat_message_hash(self, stable_key: str) -> str | None:
            return self.hashes.get(stable_key)

        def append_chat_message(
            self, stable_key: str, payload: dict, payload_hash: str
        ) -> None:
            self.hashes[stable_key] = payload_hash
            self.messages[stable_key] = payload

        def record_conflict(self, conflict: dict) -> None:
            self.conflicts.append(conflict)

    store, _persistence, session, greeting = _seeded_roleplay_store()
    dataset_key = generate_dataset_key()
    repository = SyncStateRepository(tmp_path / "roleplay-sync-state.db")
    repository.set_sync_v2_profile_state(
        server_profile_id="profile-1",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    producer = ChatSyncV2OutboxProducer(
        state_repository=repository,
        dataset_keys={"dataset-1": dataset_key},
    )
    store.sync_v2_chat_producer = producer
    store.sync_v2_server_profile_id = "profile-1"
    stable_key = f"{session.persisted_conversation_id}:{greeting.persisted_message_id}"
    baseline = producer.enqueue_chat_message(
        server_profile_id="profile-1",
        conversation_id=session.persisted_conversation_id,
        message_id=greeting.persisted_message_id,
        role="assistant",
        content="Hello User.",
    )
    baseline_envelope = baseline["outbox_entry"]["envelope"]
    store._sync_v2_message_versions[stable_key] = baseline_envelope["payload_hash"]

    plan_b = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Bravo"
    )
    assert plan_b is not None
    result_b = store.persist_roleplay_projection_plan(plan_b)
    assert (
        len(
            repository.list_pending_sync_v2_outbox_envelopes(
                server_profile_id="profile-1",
                authenticated_principal_id=None,
                workspace_scope=None,
                dataset_id="dataset-1",
            )
        )
        == 1
    )
    if execute_b:
        assert store.accept_roleplay_projection_persistence_result(result_b) is True
    plan_c = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Commander Cecelia"
    )
    assert plan_c is not None
    if not execute_b:
        assert store.accept_roleplay_projection_persistence_result(result_b) is False
    result_c = store.persist_roleplay_projection_plan(plan_c)
    assert store.accept_roleplay_projection_persistence_result(result_c) is True

    entries = repository.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="profile-1",
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id="dataset-1",
    )
    envelopes = [entry["envelope"] for entry in entries]
    assert len(envelopes) == (3 if execute_b else 2)
    for previous, current in zip(envelopes, envelopes[1:]):
        assert current["base_version"] == previous["payload_hash"]

    target = ChatTarget()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=target)
    assert [
        applier.apply(SyncV2Envelope.model_validate(envelope))["status"]
        for envelope in envelopes
    ] == ["applied"] * len(envelopes)
    assert target.messages[stable_key]["content"] == "Hello Commander Cecelia."
    assert target.conflicts == []


def test_stale_projection_after_manual_greeting_edit_never_enqueues_sync(tmp_path):
    class ChatTarget:
        def __init__(self) -> None:
            self.hashes: dict[str, str] = {}
            self.messages: dict[str, dict] = {}
            self.conflicts: list[dict] = []

        def get_chat_message_hash(self, stable_key: str) -> str | None:
            return self.hashes.get(stable_key)

        def append_chat_message(
            self, stable_key: str, payload: dict, payload_hash: str
        ) -> None:
            self.hashes[stable_key] = payload_hash
            self.messages[stable_key] = payload

        def record_conflict(self, conflict: dict) -> None:
            self.conflicts.append(conflict)

    store, _persistence, session, greeting = _seeded_roleplay_store()
    repository = SyncStateRepository(tmp_path / "manual-edit-sync-state.db")
    dataset_key = generate_dataset_key()
    repository.set_sync_v2_profile_state(
        server_profile_id="profile-1",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    producer = ChatSyncV2OutboxProducer(
        state_repository=repository,
        dataset_keys={"dataset-1": dataset_key},
    )
    store.sync_v2_chat_producer = producer
    store.sync_v2_server_profile_id = "profile-1"
    stable_key = f"{session.persisted_conversation_id}:{greeting.persisted_message_id}"
    baseline = producer.enqueue_chat_message(
        server_profile_id="profile-1",
        conversation_id=session.persisted_conversation_id,
        message_id=greeting.persisted_message_id,
        role="assistant",
        content="Hello User.",
    )
    store._sync_v2_message_versions[stable_key] = baseline["outbox_entry"]["envelope"][
        "payload_hash"
    ]

    plan_b = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Bravo"
    )
    assert plan_b is not None
    result_b = store.persist_roleplay_projection_plan(plan_b)
    store.update_message_content(greeting.id, "Manual greeting.")
    assert store.accept_roleplay_projection_persistence_result(result_b) is False

    entries = repository.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="profile-1",
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id="dataset-1",
    )
    envelopes = [entry["envelope"] for entry in entries]
    assert len(envelopes) == 2
    assert envelopes[1]["base_version"] == envelopes[0]["payload_hash"]
    target = ChatTarget()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=target)
    assert [
        applier.apply(SyncV2Envelope.model_validate(envelope))["status"]
        for envelope in envelopes
    ] == ["applied", "applied"]
    assert target.messages[stable_key]["content"] == "Manual greeting."
    assert target.conflicts == []


@pytest.mark.parametrize(
    "failed_component",
    ("system", "message"),
    ids=("system-fails", "message-fails"),
)
def test_partial_projection_failure_retains_real_durable_ancestor_for_repair(
    tmp_path, failed_component
):
    db = CharactersRAGDB(tmp_path / f"partial-{failed_component}.db", "task-5")
    service = ChatPersistenceService(db)
    conversation_id = service.create_conversation(
        assistant_kind="generic",
        assistant_id="console",
        system_prompt="Speak with Alpha.",
    )
    service.update_conversation_roleplay_context(
        conversation_id=conversation_id,
        user_name_override=None,
        character_system_template="Speak with {{user}}.",
        character_name_snapshot=None,
    )
    metadata = MessageMetadata(
        template_kind="character_greeting",
        template_source="Hello {{user}}.",
    )

    class PartialPersistence:
        def __init__(self) -> None:
            self.fail_system = failed_component == "system"
            self.fail_message = failed_component == "message"

        def create_message(self, **kwargs):
            return service.create_message(**kwargs)

        def update_conversation_system_prompt(self, **kwargs):
            if self.fail_system:
                return False
            return service.update_conversation_system_prompt(**kwargs)

        def update_message_content(self, **kwargs):
            if self.fail_message:
                return False
            return service.update_message_content(**kwargs)

    persistence = PartialPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", system_prompt="Speak with Alpha."
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = conversation_id
    session.character_system_template = "Speak with {{user}}."
    greeting = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Hello Alpha.",
        persist=True,
        metadata=metadata,
    )
    persisted_message_id = greeting.persisted_message_id
    assert persisted_message_id is not None

    plan_b = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Bravo"
    )
    assert plan_b is not None
    result_b = store.persist_roleplay_projection_plan(plan_b)
    assert result_b.persisted is False
    assert store.accept_roleplay_projection_persistence_result(result_b) is True
    durable_b_system = db.get_conversation_by_id(conversation_id)["system_prompt"]
    durable_b_message = db.get_message_by_id(persisted_message_id)["content"]
    assert (durable_b_system, durable_b_message) == (
        ("Speak with Alpha.", "Hello Bravo.")
        if failed_component == "system"
        else ("Speak with Bravo.", "Hello Alpha.")
    )

    persistence.fail_system = False
    persistence.fail_message = False
    plan_c = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Cecelia"
    )
    assert plan_c is not None
    result_c = store.persist_roleplay_projection_plan(plan_c)
    assert result_c.persisted is True
    assert store.accept_roleplay_projection_persistence_result(result_c) is True
    assert db.get_conversation_by_id(conversation_id)["system_prompt"] == (
        "Speak with Cecelia."
    )
    assert db.get_message_by_id(persisted_message_id)["content"] == ("Hello Cecelia.")
    db.close_connection()


def test_stale_refresh_keeps_live_projection_when_durable_write_refuses_or_raises():
    class RefusingPersistence(FakePersistence):
        def update_message_content(self, **kwargs):
            return False

    store, _unused, session, greeting = _seeded_roleplay_store()
    store.persistence = RefusingPersistence()
    assert (
        store.refresh_session_roleplay_projections(session.id, global_default="Rowan")
        is False
    )
    assert store.get_message(greeting.id).content == "Hello Rowan."

    class RaisingPersistence(FakePersistence):
        def update_message_content(self, **kwargs):
            raise RuntimeError("locked")

    store, _unused, session, greeting = _seeded_roleplay_store()
    store.persistence = RaisingPersistence()
    assert (
        store.refresh_session_roleplay_projections(session.id, global_default="Rowan")
        is False
    )
    assert store.get_message(greeting.id).content == "Hello Rowan."


def test_failed_regeneration_restores_greeting_provenance():
    store, _persistence, _session, greeting = _seeded_roleplay_store()

    store.begin_variant_stream(greeting.id)
    restored = store.mark_message_failed(greeting.id)

    assert restored.metadata is not None
    assert restored.metadata.template_kind == "character_greeting"


def test_presentation_context_resolves_override_identity_and_roleplay_row():
    store, _persistence, session, greeting = _seeded_roleplay_store()
    session.user_display_name_override = "Captain Rowan"
    session.identity_revision = 9

    context = store.presentation_context(session.id, "Global User")
    presentation = resolve_console_message_presentation(greeting, context)

    assert context.user_name == "Captain Rowan"
    assert context.assistant_kind == "character"
    assert context.character_name == "Alraune"
    assert context.revision == 9
    assert presentation.row_class == "console-transcript-message-roleplay-character"


def test_atomic_promotion_adapter_failure_preserves_ephemeral_fake_state():
    class FailingAtomicPersistence(FakePersistence):
        def promote_console_conversation_bundle(self, **kwargs):
            raise RuntimeError("atomic bundle failure")

    persistence = FailingAtomicPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(ephemeral=True)
    session.user_display_name_override = "Rowan"

    with pytest.raises(RuntimeError, match="atomic bundle failure"):
        store.promote_ephemeral_session(session.id)

    assert persistence.created_conversations == []
    assert session.ephemeral is True
