import asyncio
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from itertools import combinations
from typing import get_args

import pytest

from tldw_chatbook.Chat import console_chat_fork
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_fork import (
    CONSOLE_FORK_FINGERPRINT_JSON_MAX_BYTES,
    CONSOLE_FORK_TITLE_MAX_LENGTH,
    ConsoleChatForkSnapshot,
    ConsoleForkCitationState,
    ConsoleForkCitationLink,
    ConsoleForkConfigurationSnapshot,
    ConsoleForkDurability,
    ConsoleForkEligibility,
    ConsoleForkFence,
    ConsoleForkImageSelectionFence,
    ConsoleForkLineageFence,
    ConsoleForkProjectedAttachment,
    ConsoleForkProjectedGeneration,
    ConsoleForkProjectedMessage,
    default_fork_title,
    normalize_fork_title,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleVariant,
    GenerationVariantMeta,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    sanitize_fork_project_instruction_state,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class _ForkVersionPersistence:
    db = None

    def __init__(self) -> None:
        self.conversation_version = 7
        self.message_versions: dict[str, int] = {}

    def get_conversation_version(self, _conversation_id: str) -> int:
        return self.conversation_version

    def get_message_version(self, message_id: str) -> int | None:
        return self.message_versions.get(message_id)


def _fork_store(*, durable: bool = False, ephemeral: bool = False):
    persistence = _ForkVersionPersistence()
    store = ConsoleChatStore(persistence=persistence if durable else None)
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-test",
        system_prompt="Be exact.",
        pinned_prefill="source-only",
    )
    session = store.create_session(
        title="Source chat",
        workspace_id="workspace-1",
        settings=settings,
        runtime_backend="server",
        assistant_kind="persona",
        assistant_id="persona-1",
        assistant_authority_id=None,
        persona_memory_mode="read_write",
        ephemeral=ephemeral,
        project_instruction_state=ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id="binding-1",
            working_folder_locator_fingerprint="locator-1",
            project_instruction_notice_key="source-notice",
        ),
    )
    session.user_display_name_override = "Riley"
    session.character_system_template = "You are {{char}}."
    session.rag_scope_holder.set(
        RagScope(
            items=(ScopeItem("note", "7"),),
            updated_at="2026-08-27T00:00:00Z",
        )
    )
    session.speech_preferences = ConsoleSpeechPreferences(auto_speak=True)
    user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Question",
        attachments=(MessageAttachment(b"sent", "text/plain", "question.txt", 0),),
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content="Excluded activity row",
    )
    first_answer = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="First answer",
    )
    later_user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Old branch tail",
    )
    later_answer = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Old branch answer",
    )
    selected = store.create_sibling(
        first_answer.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Selected answer",
        attachments=(MessageAttachment(b"image", "image/png", "selected.png", 0),),
    )
    store.add_variant(selected.id, "Selected variant")
    selected_live = store._nodes_by_session[session.id][selected.id]
    selected_live.generation_metadata = (
        GenerationVariantMeta(
            prompt="a diagram",
            negative_prompt="",
            backend="openai",
            model="image-test",
            seed=3,
            style=None,
            params={"size": "small"},
        ),
    )
    after = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="After boundary",
    )
    for message_id, turn_id in (
        (user.id, "source-turn-1"),
        (first_answer.id, "source-turn-1"),
        (later_user.id, "source-turn-2"),
        (later_answer.id, "source-turn-2"),
        (selected.id, "source-turn-1"),
        (after.id, "source-turn-2"),
    ):
        store._nodes_by_session[session.id][message_id].turn_id = turn_id
    if durable:
        session.persisted_conversation_id = "conversation-1"
        for index, message_id in enumerate(
            (
                user.id,
                first_answer.id,
                later_user.id,
                later_answer.id,
                selected.id,
                after.id,
            ),
            start=1,
        ):
            message = store._nodes_by_session[session.id][message_id]
            message.persisted_message_id = f"persisted-{index}"
            persistence.message_versions[message.persisted_message_id] = index
        for message_id in store._nodes_by_session[session.id]:
            parent_id = store._native_parent_by_message[message_id]
            store._nodes_by_session[session.id][message_id].parent_message_id = (
                store._nodes_by_session[session.id][parent_id].persisted_message_id
                if parent_id is not None
                else None
            )
    return (
        store,
        persistence,
        session,
        user,
        first_answer,
        later_answer,
        selected,
        after,
    )


def _source_store_bytes(store: ConsoleChatStore, session_id: str) -> bytes:
    session = store._sessions[session_id]
    session_shape = (
        session.id,
        session.title,
        session.workspace_id,
        session.persisted_conversation_id,
        session.settings,
        session.context_policy_overrides,
        session.library_policy_holder.snapshot,
        session.library_policy_holder.explicitly_staged,
        session.library_policy_holder.save_pending,
        session.draft,
        session.has_user_work,
        session.pending_attachments,
        session.one_shot_prefill,
        session.rag_scope_holder.scope,
        session.runtime_backend,
        session.assistant_kind,
        session.assistant_id,
        session.assistant_authority_id,
        session.persona_memory_mode,
        session.character_id,
        session.character_name,
        session.user_display_name_override,
        session.character_system_template,
        session.speech_preferences,
        session.project_instruction_state,
        session.ephemeral,
        session.todo_store.export_snapshot(),
    )
    node_ids = tuple(store._nodes_by_session[session_id])
    return pickle.dumps(
        (
            session_shape,
            tuple(store._nodes_by_session[session_id][node_id] for node_id in node_ids),
            tuple(
                (parent, tuple(children))
                for parent, children in store._children_by_parent[session_id].items()
            ),
            tuple(
                (node_id, store._native_parent_by_message[node_id])
                for node_id in node_ids
            ),
            tuple(
                (node_id, store._message_session_index[node_id]) for node_id in node_ids
            ),
            store._active_leaf_by_session[session_id],
            tuple(message.id for message in store._messages_by_session[session_id]),
            store._context_summary_by_session[session_id],
            store._conversation_context_epochs[session_id],
            store._payload_revisions[session_id],
            tuple(
                (node_id, store._message_speech_revisions[node_id])
                for node_id in node_ids
            ),
            tuple(
                (node_id, store._message_completion_generations[node_id])
                for node_id in node_ids
            ),
            tuple(store._tool_markers_by_session.get(session_id, ())),
            store._dispatch_recoveries_by_session.get(session_id),
        ),
        protocol=5,
    )


def _configuration_snapshot() -> ConsoleForkConfigurationSnapshot:
    return ConsoleForkConfigurationSnapshot(
        workspace_id="global",
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
        rag_scope=None,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        library_policy=ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        ),
        runtime_backend="local",
        assistant_kind="generic",
        assistant_id="console",
        assistant_authority_id=None,
        persona_memory_mode=None,
        character_id=None,
        character_name=None,
        user_display_name_override=None,
        character_system_template=None,
        speech_preferences=ConsoleSpeechPreferences(),
        project_instruction_state=ProjectInstructionControlState.new_session(),
    )


_FORK_IDENTITY_FIELDS = (
    "runtime_backend",
    "assistant_kind",
    "assistant_id",
    "assistant_authority_id",
    "persona_memory_mode",
    "character_id",
)

_CANONICAL_FORK_IDENTITIES = (
    pytest.param("local", None, None, None, None, None, id="unscoped-local"),
    pytest.param("server", None, None, None, None, None, id="unscoped-server"),
    pytest.param("local", "generic", None, None, None, None, id="generic"),
    pytest.param("server", "generic", "assistant-1", None, None, None, id="generic-id"),
    pytest.param(
        "local", "persona", "persona-1", None, "read_only", None, id="persona"
    ),
    pytest.param(
        "server",
        "persona",
        "persona-1",
        None,
        "read_write",
        None,
        id="persona-server",
    ),
    pytest.param("local", "character", "7", None, None, 7, id="local-character"),
    pytest.param(
        "server",
        "character",
        "character-7",
        "catalog-1",
        None,
        None,
        id="server-character",
    ),
)

_INVALID_FORK_IDENTITIES = (
    pytest.param("chat", "generic", None, None, None, None, id="runtime"),
    pytest.param("local", "Persona", "p-1", None, None, None, id="kind-case"),
    pytest.param("local", "generic", " ", None, None, None, id="blank-id"),
    pytest.param("local", "generic", " id ", None, None, None, id="padded-id"),
    pytest.param("local", "character", "7", " ", None, 7, id="blank-authority"),
    pytest.param(
        "local",
        "character",
        "7",
        " authority ",
        None,
        7,
        id="padded-authority",
    ),
    pytest.param("local", "persona", None, None, None, None, id="persona-no-id"),
    pytest.param(
        "local",
        "persona",
        "p-1",
        "authority",
        None,
        None,
        id="persona-authority",
    ),
    pytest.param(
        "local",
        "persona",
        "p-1",
        None,
        "conversation",
        None,
        id="persona-memory",
    ),
    pytest.param("local", "persona", "p-1", None, None, 1, id="persona-character-id"),
    pytest.param(
        "local", "generic", None, None, "read_only", None, id="generic-memory"
    ),
    pytest.param(
        "local",
        "generic",
        None,
        "authority",
        None,
        None,
        id="generic-authority",
    ),
    pytest.param("local", "generic", None, None, None, 1, id="generic-character"),
    pytest.param("local", None, "assistant", None, None, None, id="null-id"),
    pytest.param(
        "local",
        "character",
        "7",
        None,
        "read_only",
        7,
        id="character-memory",
    ),
    pytest.param(
        "local",
        "character",
        "7",
        None,
        None,
        None,
        id="local-character-no-character-id",
    ),
    pytest.param(
        "local",
        "character",
        "8",
        None,
        None,
        7,
        id="local-character-mismatch",
    ),
    pytest.param(
        "local",
        "character",
        "1",
        None,
        None,
        True,
        id="boolean-character-id",
    ),
    pytest.param(
        "local",
        "character",
        "0",
        None,
        None,
        0,
        id="nonpositive-character-id",
    ),
    pytest.param(
        "server",
        "character",
        "character-7",
        None,
        None,
        7,
        id="server-character-local-id",
    ),
    pytest.param(
        "server",
        "character",
        None,
        None,
        None,
        None,
        id="server-character-no-id",
    ),
    pytest.param(
        "server",
        "generic",
        "a" * 257,
        None,
        None,
        None,
        id="oversize-assistant-id",
    ),
    pytest.param(
        "server",
        "character",
        "character-7",
        "a" * 255 + "é",
        None,
        None,
        id="oversize-multibyte-authority-id",
    ),
)


def _registration_snapshot(
    collision: tuple[str, str] | None = None,
) -> ConsoleChatForkSnapshot:
    ids = {
        "session": "fork-session",
        "conversation": "fork-conversation",
        "native": "fork-native-1",
        "persisted": "fork-persisted-1",
        "turn": "fork-turn",
        "variant": "fork-variant",
    }
    if collision is not None:
        ids[collision[0]] = ids[collision[1]] = "colliding-id"
    messages = (
        ConsoleForkProjectedMessage(
            source_native_message_id="source-native-1",
            source_persisted_message_id=None,
            source_persisted_revision=None,
            native_message_id=ids["native"],
            persisted_message_id=ids["persisted"],
            native_parent_id=None,
            persisted_parent_id=None,
            turn_id=ids["turn"],
            visible_variant_id=None,
            role=ConsoleMessageRole.USER,
            status="complete",
            content="Question",
        ),
        ConsoleForkProjectedMessage(
            source_native_message_id="source-native-2",
            source_persisted_message_id=None,
            source_persisted_revision=None,
            native_message_id="fork-native-2",
            persisted_message_id="fork-persisted-2",
            native_parent_id=ids["native"],
            persisted_parent_id=ids["persisted"],
            turn_id=ids["turn"],
            visible_variant_id=ids["variant"],
            role=ConsoleMessageRole.ASSISTANT,
            status="complete",
            content="Answer",
        ),
    )
    return ConsoleChatForkSnapshot(
        fork_session_id=ids["session"],
        fork_conversation_id=ids["conversation"],
        title="Independent fork",
        source_session_id="source-session",
        source_conversation_id=None,
        source_boundary_persisted_message_id=None,
        durable=True,
        messages=messages,
        configuration=_configuration_snapshot(),
        citation_links=(),
    )


def test_fork_titles_use_the_approved_defaults_and_bound() -> None:
    assert default_fork_title("Research notes") == "Forked from Research notes"
    assert default_fork_title("") == "Untitled chat — fork"
    assert len(normalize_fork_title("x" * 100)) == 60


def test_fork_title_rejects_blank_normalized_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        normalize_fork_title(" \n\t ")


def test_fork_title_reuses_the_console_title_deriver(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_derive(draft: str, *, max_length: int) -> str:
        calls.append((draft, max_length))
        return "canonical title"

    monkeypatch.setattr(
        console_chat_fork,
        "derive_console_session_title",
        fake_derive,
    )

    assert normalize_fork_title("  proposed title  ") == "canonical title"
    assert calls == [("  proposed title  ", CONSOLE_FORK_TITLE_MAX_LENGTH)]


def test_fork_project_instruction_state_clears_only_source_notice_authority() -> None:
    source = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-1",
        working_folder_locator_fingerprint="locator-fingerprint",
        project_instruction_notice_key="source-consent",
    )

    assert sanitize_fork_project_instruction_state(source) == replace(
        source,
        project_instruction_notice_key=None,
    )


def test_fork_project_instruction_sanitizer_rejects_nested_authority_state() -> None:
    source = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id={"permission_state": "allow"},  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="working_folder_binding_id"):
        sanitize_fork_project_instruction_state(source)


def test_fork_project_instruction_sanitizer_rejects_cycles() -> None:
    cycle: dict[str, object] = {}
    cycle["scratch_state"] = cycle
    source = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_locator_fingerprint=cycle,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="working_folder_locator_fingerprint"):
        sanitize_fork_project_instruction_state(source)


def test_fork_project_instruction_sanitizer_requires_an_exact_boolean() -> None:
    source = ProjectInstructionControlState(
        project_instructions_enabled=1,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="project_instructions_enabled"):
        sanitize_fork_project_instruction_state(source)


def test_fork_records_are_frozen_slotted_contracts() -> None:
    record_types = (
        ConsoleForkEligibility,
        ConsoleForkLineageFence,
        ConsoleForkImageSelectionFence,
        ConsoleForkFence,
        ConsoleForkProjectedMessage,
        ConsoleForkProjectedAttachment,
        ConsoleForkProjectedGeneration,
        ConsoleForkConfigurationSnapshot,
        ConsoleForkCitationLink,
        ConsoleChatForkSnapshot,
    )

    assert all(record_type.__dataclass_params__.frozen for record_type in record_types)
    assert all(hasattr(record_type, "__slots__") for record_type in record_types)

    eligibility = ConsoleForkEligibility(True)
    with pytest.raises(FrozenInstanceError):
        eligibility.eligible = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("record_type", "expected_fields"),
    (
        (ConsoleForkEligibility, ("eligible", "reason")),
        (
            ConsoleForkLineageFence,
            (
                "native_message_id",
                "persisted_message_id",
                "native_parent_id",
                "turn_id",
                "role",
                "status",
                "visible_content",
                "visible_variant_id",
                "sibling_identity",
                "persisted_revision",
                "attachment_fingerprint",
            ),
        ),
        (
            ConsoleForkImageSelectionFence,
            (
                "native_message_id",
                "selected_position",
                "browse_revision",
                "attachment_meta_fingerprint",
            ),
        ),
        (
            ConsoleForkFence,
            (
                "source_session_id",
                "source_conversation_id",
                "source_conversation_version",
                "source_durability",
                "source_title",
                "source_configuration_fingerprint",
                "boundary_message_id",
                "lineage",
                "image_selections",
            ),
        ),
        (
            ConsoleForkProjectedMessage,
            (
                "source_native_message_id",
                "source_persisted_message_id",
                "source_persisted_revision",
                "native_message_id",
                "persisted_message_id",
                "native_parent_id",
                "persisted_parent_id",
                "turn_id",
                "visible_variant_id",
                "role",
                "status",
                "content",
                "attachments",
                "generation_metadata",
            ),
        ),
        (
            ConsoleForkProjectedAttachment,
            (
                "owner_native_message_id",
                "owner_persisted_message_id",
                "position",
                "data",
                "mime_type",
                "display_name",
            ),
        ),
        (
            ConsoleForkProjectedGeneration,
            (
                "owner_native_message_id",
                "owner_persisted_message_id",
                "position",
                "prompt",
                "negative_prompt",
                "backend",
                "model",
                "seed",
                "style",
                "params_json",
            ),
        ),
        (
            ConsoleForkConfigurationSnapshot,
            (
                "workspace_id",
                "settings",
                "rag_scope",
                "context_policy_overrides",
                "library_policy",
                "runtime_backend",
                "assistant_kind",
                "assistant_id",
                "assistant_authority_id",
                "persona_memory_mode",
                "character_id",
                "character_name",
                "user_display_name_override",
                "character_system_template",
                "speech_preferences",
                "project_instruction_state",
            ),
        ),
        (
            ConsoleForkCitationLink,
            ("source_persisted_message_id", "source_revision", "state"),
        ),
        (
            ConsoleChatForkSnapshot,
            (
                "fork_session_id",
                "fork_conversation_id",
                "title",
                "source_session_id",
                "source_conversation_id",
                "source_boundary_persisted_message_id",
                "durable",
                "messages",
                "configuration",
                "citation_links",
            ),
        ),
    ),
)
def test_fork_public_record_fields_are_exact(record_type, expected_fields) -> None:
    assert tuple(field.name for field in fields(record_type)) == expected_fields


def test_fork_literal_domains_are_exact() -> None:
    assert get_args(ConsoleForkDurability) == (
        "temporary",
        "durable",
        "unsaved_persistable",
    )
    assert get_args(ConsoleForkCitationState) == (
        "active_required",
        "unavailable",
        "none",
    )


def test_fork_fingerprint_is_canonical_and_domain_separated() -> None:
    configuration = _configuration_snapshot()
    image_selection = ConsoleForkImageSelectionFence(
        native_message_id="message-1",
        selected_position=0,
        browse_revision=1,
        attachment_meta_fingerprint="sha256:attachment",
    )

    first = console_chat_fork.fingerprint_console_fork_configuration(configuration)
    repeated = console_chat_fork.fingerprint_console_fork_configuration(
        replace(configuration)
    )
    other_domain = console_chat_fork.fingerprint_console_fork_image_selection(
        image_selection
    )

    assert first == repeated
    assert first != other_domain
    assert len(first) == 64


def test_private_fork_hash_uses_canonical_json_and_purpose_domains() -> None:
    fingerprint = console_chat_fork._fingerprint_console_fork_payload

    assert fingerprint("test", {"alpha": 1, "beta": 2}) == fingerprint(
        "test", {"beta": 2, "alpha": 1}
    )
    assert fingerprint("test", {"alpha": 1}) != fingerprint("other", {"alpha": 1})


def test_fork_fingerprint_rejects_unbounded_allowlisted_payload() -> None:
    with pytest.raises(ValueError, match="bounded"):
        console_chat_fork.fingerprint_console_fork_configuration(
            replace(
                _configuration_snapshot(),
                workspace_id="x" * CONSOLE_FORK_FINGERPRINT_JSON_MAX_BYTES,
            )
        )


@pytest.mark.parametrize(
    "forbidden_payload",
    (
        {"scratch_owner": "source-session"},
        {"permission_state": {"fs_write": "allow"}},
        {"raw_path": "/private/tmp/source-scratch"},
    ),
)
def test_fork_fingerprint_public_flow_rejects_non_allowlisted_payloads(
    forbidden_payload,
) -> None:
    with pytest.raises(TypeError, match="ConsoleForkConfigurationSnapshot"):
        console_chat_fork.fingerprint_console_fork_configuration(forbidden_payload)


def test_fork_fingerprint_rejects_source_notice_authority() -> None:
    source_authority = replace(
        _configuration_snapshot(),
        project_instruction_state=replace(
            ProjectInstructionControlState.new_session(),
            project_instruction_notice_key="source-consent",
        ),
    )

    with pytest.raises(ValueError, match="notice authority"):
        console_chat_fork.fingerprint_console_fork_configuration(source_authority)


def test_fork_fingerprint_rejects_nested_project_authority_state() -> None:
    configuration = replace(
        _configuration_snapshot(),
        project_instruction_state=ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id={"permission_state": "allow"},  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(TypeError, match="working_folder_binding_id"):
        console_chat_fork.fingerprint_console_fork_configuration(configuration)


def test_fork_fingerprint_rejects_cycles_before_canonical_json() -> None:
    cycle: dict[str, object] = {}
    cycle["scratch_state"] = cycle
    configuration = replace(
        _configuration_snapshot(),
        workspace_id=cycle,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="workspace_id"):
        console_chat_fork.fingerprint_console_fork_configuration(configuration)


@pytest.mark.parametrize(
    "configuration",
    (
        replace(
            _configuration_snapshot(),
            settings=replace(_configuration_snapshot().settings, streaming=1),  # type: ignore[arg-type]
        ),
        replace(
            _configuration_snapshot(),
            rag_scope=RagScope(
                items=[ScopeItem("note", "7")],  # type: ignore[arg-type]
                updated_at="2026-08-26T00:00:00Z",
            ),
        ),
        replace(
            _configuration_snapshot(),
            library_policy=ConsoleLibraryPolicyCandidate(
                auto_retrieve="never",  # type: ignore[arg-type]
                assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            ),
        ),
    ),
)
def test_fork_configuration_fingerprint_rejects_malformed_nested_leaf_types(
    configuration,
) -> None:
    with pytest.raises(TypeError, match="Fork configuration"):
        console_chat_fork.fingerprint_console_fork_configuration(configuration)


@pytest.mark.parametrize(_FORK_IDENTITY_FIELDS, _CANONICAL_FORK_IDENTITIES)
def test_fork_configuration_accepts_only_canonical_persistence_identities(
    runtime_backend,
    assistant_kind,
    assistant_id,
    assistant_authority_id,
    persona_memory_mode,
    character_id,
) -> None:
    configuration = replace(
        _configuration_snapshot(),
        runtime_backend=runtime_backend,
        assistant_kind=assistant_kind,
        assistant_id=assistant_id,
        assistant_authority_id=assistant_authority_id,
        persona_memory_mode=persona_memory_mode,
        character_id=character_id,
    )

    assert console_chat_fork.fingerprint_console_fork_configuration(configuration)


@pytest.mark.parametrize(_FORK_IDENTITY_FIELDS, _INVALID_FORK_IDENTITIES)
def test_invalid_persistence_identity_is_rejected_at_every_fork_boundary(
    runtime_backend,
    assistant_kind,
    assistant_id,
    assistant_authority_id,
    persona_memory_mode,
    character_id,
) -> None:
    identity = dict(
        zip(
            _FORK_IDENTITY_FIELDS,
            (
                runtime_backend,
                assistant_kind,
                assistant_id,
                assistant_authority_id,
                persona_memory_mode,
                character_id,
            ),
            strict=True,
        )
    )
    configuration = replace(_configuration_snapshot(), **identity)
    with pytest.raises((TypeError, ValueError)):
        console_chat_fork.fingerprint_console_fork_configuration(configuration)

    source_store, _, session, _, _, _, selected, _ = _fork_store()
    for field_name, value in identity.items():
        setattr(session, field_name, value)
    assert source_store.fork_eligibility(selected.id).eligible is False
    with pytest.raises(ValueError):
        source_store.issue_fork_fence(selected.id)

    registration_store = ConsoleChatStore()
    snapshot = replace(
        _registration_snapshot(),
        configuration=configuration,
    )
    with pytest.raises((TypeError, ValueError)):
        registration_store.register_fork_snapshot(snapshot, activate=False)
    assert registration_store.sessions() == []


def test_local_character_fork_rejects_mismatched_destination_authority(
    tmp_path,
) -> None:
    db = CharactersRAGDB(
        tmp_path / "fork-authority.sqlite",
        client_id="fork-authority",
    )
    try:
        service = ChatPersistenceService(db)
        character_id = db.add_character_card({"name": "Local character"})
        local_authority = db.get_local_authority_id()
        mismatched_authority = f"{local_authority}-mismatch"
        store = ConsoleChatStore(persistence=service)
        settings = ConsoleSessionSettings(provider="openai", model="gpt-test")

        invalid_source = store.create_session(
            title="Invalid source",
            settings=settings,
            runtime_backend="local",
            assistant_kind="character",
            assistant_id=str(character_id),
            assistant_authority_id=mismatched_authority,
            character_id=character_id,
        )
        store.append_message(
            invalid_source.id,
            role=ConsoleMessageRole.USER,
            content="Question",
        )
        invalid_boundary = store.append_message(
            invalid_source.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Answer",
        )

        assert store.fork_eligibility(invalid_boundary.id).eligible is False
        with pytest.raises(ValueError):
            store.issue_fork_fence(invalid_boundary.id)

        valid_source = store.create_session(
            title="Valid source",
            settings=settings,
            runtime_backend="local",
            assistant_kind="character",
            assistant_id=str(character_id),
            assistant_authority_id=local_authority,
            character_id=character_id,
        )
        store.append_message(
            valid_source.id,
            role=ConsoleMessageRole.USER,
            content="Question",
        )
        valid_boundary = store.append_message(
            valid_source.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Answer",
        )
        fence = store.issue_fork_fence(valid_boundary.id)
        valid_source.assistant_authority_id = mismatched_authority

        with pytest.raises(ValueError, match="source changed"):
            store.stage_fork_snapshot(
                fence,
                title="Independent fork",
                fork_session_id="staged-fork-session",
                fork_conversation_id="staged-fork-conversation",
            )

        invalid_snapshot = replace(
            _registration_snapshot(),
            configuration=replace(
                _configuration_snapshot(),
                runtime_backend="local",
                assistant_kind="character",
                assistant_id=str(character_id),
                assistant_authority_id=mismatched_authority,
                persona_memory_mode=None,
                character_id=character_id,
            ),
        )
        with pytest.raises(ValueError):
            store.register_fork_snapshot(invalid_snapshot, activate=False)
        assert "fork-session" not in {session.id for session in store.sessions()}
    finally:
        db.close_connection()


def test_local_character_fork_accepts_matching_destination_authority(tmp_path) -> None:
    db = CharactersRAGDB(
        tmp_path / "matching-fork-authority.sqlite",
        client_id="matching-fork-authority",
    )
    try:
        service = ChatPersistenceService(db)
        character_id = db.add_character_card({"name": "Local character"})
        local_authority = db.get_local_authority_id()
        store = ConsoleChatStore(persistence=service)
        source = store.create_session(
            title="Valid source",
            settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
            runtime_backend="local",
            assistant_kind="character",
            assistant_id=str(character_id),
            assistant_authority_id=local_authority,
            character_id=character_id,
        )
        store.append_message(
            source.id,
            role=ConsoleMessageRole.USER,
            content="Question",
        )
        boundary = store.append_message(
            source.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Answer",
        )

        snapshot = store.stage_fork_snapshot(
            store.issue_fork_fence(boundary.id),
            title="Independent fork",
            fork_session_id="fork-session",
            fork_conversation_id="fork-conversation",
        )
        fork_session = store.register_fork_snapshot(snapshot, activate=False)

        assert fork_session.assistant_authority_id == local_authority
        assert fork_session.persisted_conversation_id == "fork-conversation"
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "image_selection",
    (
        ConsoleForkImageSelectionFence(
            native_message_id="message-1",
            selected_position=True,  # type: ignore[arg-type]
            browse_revision=1,
            attachment_meta_fingerprint="sha256:attachment",
        ),
        ConsoleForkImageSelectionFence(
            native_message_id="message-1",
            selected_position=0,
            browse_revision=1,
            attachment_meta_fingerprint={"raw_path": "/tmp/image"},  # type: ignore[arg-type]
        ),
    ),
)
def test_fork_image_selection_fingerprint_rejects_malformed_leaf_types(
    image_selection,
) -> None:
    with pytest.raises(TypeError, match="Fork image selection"):
        console_chat_fork.fingerprint_console_fork_image_selection(image_selection)


def test_generic_fork_payload_fingerprint_is_not_public() -> None:
    assert not hasattr(console_chat_fork, "fingerprint_console_fork_payload")


def test_issue_fork_fence_uses_only_the_canonical_active_prefix() -> None:
    store, _, session, user, first_answer, later_answer, selected, after = _fork_store()
    active_before = store.active_path_message_ids(session.id)
    selected_variant_before = store.get_message(selected.id).variants.current.id

    fence = store.issue_fork_fence(selected.id)

    assert [entry.native_message_id for entry in fence.lineage] == [
        user.id,
        selected.id,
    ]
    assert fence.boundary_message_id == selected.id
    assert fence.lineage[-1].visible_content == "Selected variant"
    assert fence.lineage[-1].visible_variant_id == selected_variant_before
    assert first_answer.id not in {entry.native_message_id for entry in fence.lineage}
    assert later_answer.id not in {entry.native_message_id for entry in fence.lineage}
    assert after.id not in {entry.native_message_id for entry in fence.lineage}
    assert all(entry.role is not ConsoleMessageRole.TOOL for entry in fence.lineage)
    assert store.active_path_message_ids(session.id) == active_before
    assert store.get_message(selected.id).variants.current.id == selected_variant_before


def test_issue_fork_fence_captures_the_exact_image_selection_tuple() -> None:
    store, _, _, _, _, _, selected, _ = _fork_store()
    selection = ConsoleForkImageSelectionFence(
        native_message_id=selected.id,
        selected_position=0,
        browse_revision=7,
        attachment_meta_fingerprint="sha256:selected-image",
    )

    fence = store.issue_fork_fence(
        selected.id,
        image_selections=(selection,),
    )

    assert fence.image_selections == (selection,)
    assert (
        store.validate_fork_fence(
            fence,
            image_selections=(selection,),
        )
        is True
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("selected_position", 1),
        ("browse_revision", 8),
        ("attachment_meta_fingerprint", "sha256:changed-image"),
    ),
)
def test_validate_fork_fence_rejects_each_changed_image_selection_field(
    field_name,
    changed_value,
) -> None:
    store, _, _, _, _, _, selected, _ = _fork_store()
    selection = ConsoleForkImageSelectionFence(
        native_message_id=selected.id,
        selected_position=0,
        browse_revision=7,
        attachment_meta_fingerprint="sha256:selected-image",
    )
    fence = store.issue_fork_fence(
        selected.id,
        image_selections=(selection,),
    )

    changed = replace(selection, **{field_name: changed_value})

    assert (
        store.validate_fork_fence(
            fence,
            image_selections=(changed,),
        )
        is False
    )


def test_selected_image_fence_fails_closed_when_attachment_is_removed() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    selection = ConsoleForkImageSelectionFence(
        native_message_id=selected.id,
        selected_position=0,
        browse_revision=7,
        attachment_meta_fingerprint="sha256:selected-image",
    )
    fence = store.issue_fork_fence(
        selected.id,
        image_selections=(selection,),
    )
    store._nodes_by_session[session.id][selected.id].attachments = ()

    assert store.validate_fork_fence(fence, image_selections=(selection,)) is False
    with pytest.raises(ValueError, match="^Console fork source changed\\.$"):
        store.stage_fork_snapshot(
            fence,
            title="Independent fork",
            fork_session_id="fork-session",
            fork_conversation_id="fork-conversation",
        )


def test_selected_image_fence_fails_closed_when_generation_is_removed() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    selection = ConsoleForkImageSelectionFence(
        native_message_id=selected.id,
        selected_position=0,
        browse_revision=7,
        attachment_meta_fingerprint="sha256:selected-image",
    )
    fence = store.issue_fork_fence(
        selected.id,
        image_selections=(selection,),
    )
    store._nodes_by_session[session.id][selected.id].generation_metadata = ()

    assert store.validate_fork_fence(fence, image_selections=(selection,)) is False
    with pytest.raises(ValueError, match="^Console fork source changed\\.$"):
        store.stage_fork_snapshot(
            fence,
            title="Independent fork",
            fork_session_id="fork-session",
            fork_conversation_id="fork-conversation",
        )


def test_fork_fence_rejects_a_boundary_outside_the_active_path() -> None:
    store, _, _, _, first_answer, _, _, _ = _fork_store()

    assert store.fork_eligibility(first_answer.id).eligible is False
    with pytest.raises(ValueError, match="active path"):
        store.issue_fork_fence(first_answer.id)


@pytest.mark.parametrize(
    ("role", "status", "content", "eligible"),
    (
        (ConsoleMessageRole.USER, "complete", "sent", True),
        (ConsoleMessageRole.ASSISTANT, "complete", "answer", True),
        (ConsoleMessageRole.ASSISTANT, "stopped", "partial", True),
        (ConsoleMessageRole.ASSISTANT, "failed", "partial", True),
        (ConsoleMessageRole.USER, "pending", "draft", False),
        (ConsoleMessageRole.USER, "stopped", "sent", False),
        (ConsoleMessageRole.ASSISTANT, "streaming", "partial", False),
        (ConsoleMessageRole.ASSISTANT, "Complete", "answer", False),
        (ConsoleMessageRole.ASSISTANT, "discarded", "answer", False),
        (ConsoleMessageRole.ASSISTANT, "failed", "", False),
        (ConsoleMessageRole.TOOL, "complete", "tool", False),
    ),
)
def test_fork_eligibility_requires_stable_user_or_assistant_content(
    role, status, content, eligible
) -> None:
    store = ConsoleChatStore()
    session = store.create_session(settings=ConsoleSessionSettings(provider="openai"))
    message = store.append_message(session.id, role=role, content=content)
    if role is not ConsoleMessageRole.TOOL:
        store._nodes_by_session[session.id][message.id].status = status

    result = store.fork_eligibility(message.id)

    assert result.eligible is eligible
    assert bool(result.reason) is not eligible


@pytest.mark.parametrize(
    "canonical_role",
    (ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT),
)
def test_fork_eligibility_rejects_raw_roles_that_compare_equal_to_the_enum(
    canonical_role,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session(settings=ConsoleSessionSettings(provider="openai"))
    message = store.append_message(
        session.id,
        role=canonical_role,
        content="stable content",
    )
    raw_role = canonical_role.value
    assert raw_role == canonical_role
    store._nodes_by_session[session.id][message.id].role = raw_role  # type: ignore[assignment]

    assert store.fork_eligibility(message.id).eligible is False
    with pytest.raises(ValueError):
        store.issue_fork_fence(message.id)


def test_validate_fork_fence_rejects_a_raw_role_after_issue() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    fence = store.issue_fork_fence(selected.id)
    selected_live = store._nodes_by_session[session.id][selected.id]
    selected_live.role = selected_live.role.value  # type: ignore[assignment]

    assert store.validate_fork_fence(fence) is False


def test_fork_registration_rejects_raw_roles_without_publishing() -> None:
    store = ConsoleChatStore()
    snapshot = _registration_snapshot()
    first, *rest = snapshot.messages
    snapshot = replace(
        snapshot,
        messages=(replace(first, role="user"), *rest),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="state"):
        store.register_fork_snapshot(snapshot, activate=False)

    assert store.sessions() == []
    assert store._message_session_index == {}


def test_fork_boundaries_reject_a_string_status_subclass() -> None:
    class StatusText(str):
        pass

    store = ConsoleChatStore()
    session = store.create_session(settings=ConsoleSessionSettings(provider="openai"))
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="stable content",
    )
    store._nodes_by_session[session.id][message.id].status = StatusText("complete")

    assert store.fork_eligibility(message.id).eligible is False
    with pytest.raises(ValueError):
        store.issue_fork_fence(message.id)

    registration_store = ConsoleChatStore()
    snapshot = _registration_snapshot()
    first, *rest = snapshot.messages
    snapshot = replace(
        snapshot,
        messages=(replace(first, status=StatusText("complete")), *rest),
    )
    with pytest.raises(ValueError, match="state"):
        registration_store.register_fork_snapshot(snapshot, activate=False)
    assert registration_store.sessions() == []


def test_durable_fence_requires_every_prefix_message_to_be_persisted() -> None:
    store, _, session, user, _, _, selected, _ = _fork_store(durable=True)
    store._nodes_by_session[session.id][user.id].persisted_message_id = None

    assert store.fork_eligibility(selected.id).eligible is False
    with pytest.raises(ValueError, match="saved"):
        store.issue_fork_fence(selected.id)


@pytest.mark.parametrize(
    "mutation",
    (
        "configuration",
        "conversation_version",
        "durability",
        "title",
        "content",
        "status",
        "turn",
        "parent",
        "persisted_parent",
        "selected_variant",
        "siblings",
        "attachment",
        "generation",
        "persisted_id",
        "persisted_version",
    ),
)
def test_validate_fork_fence_rechecks_every_captured_source_field(mutation) -> None:
    store, persistence, session, user, first_answer, _, selected, _ = _fork_store(
        durable=True
    )
    fence = store.issue_fork_fence(selected.id)
    selected_live = store._nodes_by_session[session.id][selected.id]

    if mutation == "configuration":
        session.user_display_name_override = "Changed"
    elif mutation == "conversation_version":
        persistence.conversation_version += 1
    elif mutation == "durability":
        session.persisted_conversation_id = None
    elif mutation == "title":
        session.title = "Renamed"
    elif mutation == "content":
        store._nodes_by_session[session.id][user.id].content = "Changed"
    elif mutation == "status":
        selected_live.status = "stopped"
    elif mutation == "turn":
        selected_live.turn_id = "changed-source-turn"
    elif mutation == "parent":
        store._native_parent_by_message[selected.id] = None
    elif mutation == "persisted_parent":
        selected_live.parent_message_id = None
    elif mutation == "selected_variant":
        selected_live.variants.variants.append(
            ConsoleVariant(
                content=selected_live.content,
                id="same-content-other-variant",
            )
        )
        selected_live.variants.selected_index = len(selected_live.variants.variants) - 1
    elif mutation == "siblings":
        store._children_by_parent[session.id][user.id].remove(first_answer.id)
    elif mutation == "attachment":
        selected_live.attachments = (
            MessageAttachment(b"changed", "image/png", "selected.png", 0),
        )
    elif mutation == "generation":
        selected_live.generation_metadata = (
            replace(selected_live.generation_metadata[0], prompt="changed"),
        )
    elif mutation == "persisted_id":
        selected_live.persisted_message_id = "other-id"
        persistence.message_versions["other-id"] = fence.lineage[-1].persisted_revision
    else:
        persistence.message_versions[selected_live.persisted_message_id] += 1

    if mutation in {"status", "turn", "selected_variant", "persisted_id"}:
        assert store.fork_eligibility(selected.id).eligible is True
    assert store.validate_fork_fence(fence) is False


@pytest.mark.parametrize(
    "field_name",
    (
        "workspace_id",
        "settings",
        "rag_scope",
        "context_policy_overrides",
        "library_policy",
        "runtime_backend",
        "assistant_kind",
        "assistant_id",
        "assistant_authority_id",
        "persona_memory_mode",
        "character_id",
        "character_name",
        "user_display_name_override",
        "character_system_template",
        "speech_preferences",
        "project_instruction_state",
    ),
)
def test_every_allowlisted_configuration_field_stales_the_fence(field_name) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    fence = store.issue_fork_fence(selected.id)

    if field_name == "workspace_id":
        session.workspace_id = "workspace-2"
    elif field_name == "settings":
        session.settings = replace(session.settings, model="changed-model")
    elif field_name == "rag_scope":
        session.rag_scope_holder.set(None)
    elif field_name == "context_policy_overrides":
        session.context_policy_overrides = ConsoleContextPolicyOverrides(
            custom_budget_tokens=2048
        )
    elif field_name == "library_policy":
        session.library_policy_holder.snapshot = replace(
            session.library_policy_holder.snapshot,
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
        )
    elif field_name == "speech_preferences":
        session.speech_preferences = ConsoleSpeechPreferences(auto_speak=False)
    elif field_name == "project_instruction_state":
        session.project_instruction_state = replace(
            session.project_instruction_state,
            working_folder_binding_id="binding-2",
        )
    else:
        replacement_values = {
            "runtime_backend": "local",
            "assistant_kind": "character",
            "assistant_id": "persona-2",
            "assistant_authority_id": "authority-2",
            "persona_memory_mode": "read_only",
            "character_id": 7,
            "character_name": "Alba",
            "user_display_name_override": "Morgan",
            "character_system_template": "Changed template",
        }
        setattr(session, field_name, replacement_values[field_name])

    assert store.validate_fork_fence(fence) is False


def test_unrelated_excluded_state_does_not_stale_or_enter_the_snapshot() -> None:
    store, _, session, _, first_answer, _, selected, after = _fork_store()
    fence = store.issue_fork_fence(selected.id)
    store.set_session_draft(session.id, "excluded draft")
    store.set_session_one_shot_prefill(session.id, "excluded prefill")
    session.settings = replace(session.settings, pinned_prefill="changed source pin")
    session.pending_attachments.append(object())  # type: ignore[arg-type]
    session.scratch_authority = "source-scratch"  # type: ignore[attr-defined]
    session.approval_authority = "source-approval"  # type: ignore[attr-defined]
    session.run_state = "source-run"  # type: ignore[attr-defined]
    session.presentation_state = "source-presentation"  # type: ignore[attr-defined]
    selected_live = store._nodes_by_session[session.id][selected.id]
    selected_live.usage = object()  # type: ignore[assignment]
    selected_live.provider_continuation = object()  # type: ignore[assignment]
    selected_live.activity_presentation = object()  # type: ignore[assignment]
    selected_live.video_metadata = object()  # type: ignore[assignment]
    selected_live.live_activity = "excluded live activity"
    store._nodes_by_session[session.id][first_answer.id].content = "changed off path"
    store._nodes_by_session[session.id][after.id].content = "changed after boundary"
    store._context_summary_by_session[session.id] = ("excluded summary", selected.id)
    store._dispatch_recoveries_by_session[session.id] = object()  # type: ignore[assignment]
    session.todo_store.create(content="excluded todo")

    assert store.validate_fork_fence(fence) is True
    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )

    assert snapshot.configuration.settings.pinned_prefill is None
    fork_session = store.register_fork_snapshot(snapshot, activate=False)
    fork_message = store.get_message(snapshot.messages[-1].native_message_id)
    assert fork_session.draft == ""
    assert fork_session.pending_attachments == []
    assert fork_session.one_shot_prefill is None
    assert fork_session.todo_store.list_after(None) == []
    assert not hasattr(fork_session, "scratch_authority")
    assert not hasattr(fork_session, "approval_authority")
    assert not hasattr(fork_session, "run_state")
    assert not hasattr(fork_session, "presentation_state")
    assert fork_message.usage is None
    assert fork_message.provider_continuation is None
    assert fork_message.activity_presentation is None
    assert fork_message.video_metadata is None
    assert fork_message.live_activity == ""
    assert store.session_context_summary(fork_session.id) == (None, None)
    assert store.dispatch_recovery_for_session(fork_session.id) is None


@pytest.mark.parametrize("source_kind", ("durable", "unsaved", "temporary"))
def test_stage_and_register_fork_preserves_source_and_allocates_fresh_ownership(
    source_kind,
) -> None:
    durable = source_kind == "durable"
    ephemeral = source_kind == "temporary"
    store, _, session, user, _, _, selected, _ = _fork_store(
        durable=durable,
        ephemeral=ephemeral,
    )
    before = _source_store_bytes(store, session.id)
    source_messages = tuple(store._nodes_by_session[session.id].values())
    source_ids = {message.id for message in source_messages}
    source_ids.update(
        message.persisted_message_id
        for message in source_messages
        if message.persisted_message_id is not None
    )
    source_ids.update(
        message.turn_id for message in source_messages if message.turn_id is not None
    )
    source_ids.update(
        message.variants.turn_id
        for message in source_messages
        if message.variants is not None
    )
    source_variant_ids = {
        variant.id
        for message in source_messages
        if message.variants is not None
        for variant in message.variants.variants
    }
    fence = store.issue_fork_fence(selected.id)
    fork_conversation_id = None if ephemeral else "fork-conversation"

    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id=fork_conversation_id,
    )
    assert _source_store_bytes(store, session.id) == before
    fork_session = store.register_fork_snapshot(snapshot, activate=False)

    assert _source_store_bytes(store, session.id) == before
    assert fork_session.id == "fork-session"
    assert fork_session.id != session.id
    assert fork_session.ephemeral is ephemeral
    assert fork_session.persisted_conversation_id == fork_conversation_id
    assert store.active_session_id == session.id
    assert snapshot.source_conversation_id == session.persisted_conversation_id
    assert not hasattr(fork_session, "forked_from_conversation_id")
    assert not hasattr(fork_session, "forked_from_message_id")
    target_ownership_ids = {snapshot.fork_session_id}
    target_ownership_ids.update(
        value
        for message in snapshot.messages
        for value in (
            message.native_message_id,
            message.persisted_message_id,
            message.native_parent_id,
            message.persisted_parent_id,
            message.turn_id,
            message.visible_variant_id,
        )
        if value is not None
    )
    target_ownership_ids.update(
        attachment.owner_native_message_id
        for message in snapshot.messages
        for attachment in message.attachments
    )
    target_ownership_ids.update(
        attachment.owner_persisted_message_id
        for message in snapshot.messages
        for attachment in message.attachments
        if attachment.owner_persisted_message_id is not None
    )
    target_ownership_ids.update(
        generation.owner_native_message_id
        for message in snapshot.messages
        for generation in message.generation_metadata
    )
    target_ownership_ids.update(
        generation.owner_persisted_message_id
        for message in snapshot.messages
        for generation in message.generation_metadata
        if generation.owner_persisted_message_id is not None
    )
    assert source_ids.isdisjoint(target_ownership_ids)
    assert source_variant_ids.isdisjoint(target_ownership_ids)
    assert all(
        message.native_parent_id is None
        or message.native_parent_id
        in {projected.native_message_id for projected in snapshot.messages}
        for message in snapshot.messages
    )
    assert all(
        message.turn_id is None or message.turn_id not in source_ids
        for message in snapshot.messages
    )
    assert all(
        message.visible_variant_id is None
        or message.visible_variant_id not in source_variant_ids
        for message in snapshot.messages
    )
    for message in snapshot.messages:
        assert all(
            attachment.owner_native_message_id == message.native_message_id
            and attachment.owner_native_message_id != message.source_native_message_id
            for attachment in message.attachments
        )
        assert all(
            generation.owner_native_message_id == message.native_message_id
            and generation.owner_native_message_id != message.source_native_message_id
            for generation in message.generation_metadata
        )
    if ephemeral:
        assert all(
            message.persisted_message_id is None for message in snapshot.messages
        )
        assert all(message.persisted_parent_id is None for message in snapshot.messages)
    else:
        assert all(message.persisted_message_id for message in snapshot.messages)
        assert all(
            message.persisted_parent_id is None
            or message.persisted_parent_id
            in {projected.persisted_message_id for projected in snapshot.messages}
            for message in snapshot.messages
        )


@pytest.mark.parametrize(
    ("selected_turn_id", "shared_target_turn"),
    (("source-turn-1", True), ("source-turn-split", False)),
)
def test_stage_fork_snapshot_preserves_fenced_turn_grouping(
    selected_turn_id,
    shared_target_turn,
) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    store._nodes_by_session[session.id][selected.id].turn_id = selected_turn_id
    fence = store.issue_fork_fence(selected.id)

    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )

    user_turn, assistant_turn = (message.turn_id for message in snapshot.messages)
    assert user_turn is not None
    assert assistant_turn is not None
    assert (user_turn == assistant_turn) is shared_target_turn


@pytest.mark.parametrize(
    "mutation",
    ("turn", "attachment", "generation", "configuration", "role", "status"),
)
def test_stage_fork_snapshot_rejects_late_source_mutation(
    mutation,
    monkeypatch,
) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    fence = store.issue_fork_fence(selected.id)
    original_validate = store.validate_fork_fence
    calls = 0

    class StatusText(str):
        pass

    def validate_then_mutate(candidate, *, image_selections=()):
        nonlocal calls
        calls += 1
        valid = original_validate(candidate, image_selections=image_selections)
        if calls != 1:
            return valid
        selected_live = store._nodes_by_session[session.id][selected.id]
        if mutation == "turn":
            selected_live.turn_id = "late-live-turn"
        elif mutation == "attachment":
            selected_live.attachments = (
                replace(selected_live.attachments[0], data=b"late-image"),
            )
        elif mutation == "generation":
            selected_live.generation_metadata = (
                replace(selected_live.generation_metadata[0], prompt="late prompt"),
            )
        elif mutation == "configuration":
            session.user_display_name_override = "Late display name"
        elif mutation == "role":
            selected_live.role = selected_live.role.value  # type: ignore[assignment]
        else:
            selected_live.status = StatusText("complete")  # type: ignore[assignment]
        return valid

    monkeypatch.setattr(store, "validate_fork_fence", validate_then_mutate)
    before_session_ids = tuple(item.id for item in store.sessions())
    before_message_index = dict(store._message_session_index)

    with pytest.raises(ValueError, match="source changed"):
        store.stage_fork_snapshot(
            fence,
            title="Independent fork",
            fork_session_id="fork-session",
            fork_conversation_id="fork-conversation",
        )

    assert calls == 2
    assert tuple(item.id for item in store.sessions()) == before_session_ids
    assert store._message_session_index == before_message_index
    assert "fork-session" not in store._nodes_by_session


@pytest.mark.parametrize(
    ("mutation", "selected_position"),
    (
        ("attachment", None),
        ("generation", None),
        ("attachment", 0),
        ("generation", 0),
    ),
)
def test_stage_fork_snapshot_rejects_aba_payload_mutation(
    mutation,
    selected_position,
    monkeypatch,
) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    image_selections = (
        (
            ConsoleForkImageSelectionFence(
                native_message_id=selected.id,
                selected_position=selected_position,
                browse_revision=1,
                attachment_meta_fingerprint="sha256:selected-image",
            ),
        )
        if selected_position is not None
        else ()
    )
    fence = store.issue_fork_fence(
        selected.id,
        image_selections=image_selections,
    )
    selected_live = store._nodes_by_session[session.id][selected.id]
    original_attachments = selected_live.attachments
    original_generation = selected_live.generation_metadata
    original_validate = store.validate_fork_fence
    calls = 0

    def validate_around_transient_value(candidate, *, image_selections=()):
        nonlocal calls
        calls += 1
        if calls == 1:
            valid = original_validate(candidate, image_selections=image_selections)
            if mutation == "attachment":
                selected_live.attachments = (
                    replace(original_attachments[0], data=b"transient-image"),
                )
            else:
                selected_live.generation_metadata = (
                    replace(original_generation[0], prompt="transient prompt"),
                )
            return valid
        selected_live.attachments = original_attachments
        selected_live.generation_metadata = original_generation
        return original_validate(candidate, image_selections=image_selections)

    monkeypatch.setattr(
        store,
        "validate_fork_fence",
        validate_around_transient_value,
    )

    with pytest.raises(ValueError, match="source changed"):
        store.stage_fork_snapshot(
            fence,
            title="Independent fork",
            fork_session_id="fork-session",
            fork_conversation_id="fork-conversation",
        )

    assert calls == 2
    assert "fork-session" not in {item.id for item in store.sessions()}


@pytest.mark.parametrize(
    ("field_name", "transient_value"),
    (
        ("role", ConsoleMessageRole.USER),
        ("status", "stopped"),
        ("persisted_message_id", "transient-persisted-message"),
    ),
)
def test_stage_fork_snapshot_uses_fenced_source_identity_across_aba(
    field_name,
    transient_value,
    monkeypatch,
) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store(durable=True)
    fence = store.issue_fork_fence(selected.id)
    entry = fence.lineage[-1]
    selected_live = store._nodes_by_session[session.id][selected.id]
    original_value = getattr(selected_live, field_name)
    original_validate = store.validate_fork_fence
    calls = 0

    def validate_around_transient_value(candidate, *, image_selections=()):
        nonlocal calls
        calls += 1
        if calls == 1:
            valid = original_validate(candidate, image_selections=image_selections)
            setattr(selected_live, field_name, transient_value)
            return valid
        setattr(selected_live, field_name, original_value)
        return original_validate(candidate, image_selections=image_selections)

    monkeypatch.setattr(
        store,
        "validate_fork_fence",
        validate_around_transient_value,
    )

    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected = snapshot.messages[-1]

    assert calls == 2
    assert projected.source_native_message_id == entry.native_message_id
    assert projected.source_persisted_message_id == entry.persisted_message_id
    assert projected.source_persisted_revision == entry.persisted_revision
    assert projected.role is entry.role
    assert projected.status == entry.status


@pytest.mark.parametrize(
    ("selected_position", "expected_data", "expected_prompt"),
    (
        (None, (b"image", b"second-image"), ("a diagram", "second prompt")),
        (1, (b"second-image",), ("second prompt",)),
    ),
)
def test_stage_fork_snapshot_projects_the_fenced_attachment_scope(
    selected_position,
    expected_data,
    expected_prompt,
) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    selected_live = store._nodes_by_session[session.id][selected.id]
    selected_live.attachments = (
        *selected_live.attachments,
        MessageAttachment(b"second-image", "image/png", "second.png", 1),
    )
    selected_live.generation_metadata = (
        *selected_live.generation_metadata,
        GenerationVariantMeta(
            prompt="second prompt",
            negative_prompt="",
            backend="openai",
            model="image-test",
            seed=4,
            style=None,
            params={"size": "small"},
        ),
    )
    image_selections = (
        (
            ConsoleForkImageSelectionFence(
                native_message_id=selected.id,
                selected_position=selected_position,
                browse_revision=1,
                attachment_meta_fingerprint="sha256:selected-image",
            ),
        )
        if selected_position is not None
        else ()
    )
    fence = store.issue_fork_fence(
        selected.id,
        image_selections=image_selections,
    )

    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected = snapshot.messages[-1]

    assert (
        tuple(attachment.data for attachment in projected.attachments) == expected_data
    )
    assert (
        tuple(metadata.prompt for metadata in projected.generation_metadata)
        == expected_prompt
    )


def test_configuration_snapshot_is_the_exact_sanitized_allowlist() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    fence = store.issue_fork_fence(selected.id)
    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    configuration = snapshot.configuration

    assert configuration == ConsoleForkConfigurationSnapshot(
        workspace_id=session.workspace_id,
        settings=replace(session.settings, pinned_prefill=None),
        rag_scope=session.rag_scope_holder.scope,
        context_policy_overrides=session.context_policy_overrides,
        library_policy=ConsoleLibraryPolicyCandidate(
            auto_retrieve=session.library_policy_holder.snapshot.auto_retrieve,
            assistant_access=session.library_policy_holder.snapshot.assistant_access,
        ),
        runtime_backend="server",
        assistant_kind="persona",
        assistant_id="persona-1",
        assistant_authority_id=None,
        persona_memory_mode="read_write",
        character_id=None,
        character_name=None,
        user_display_name_override="Riley",
        character_system_template="You are {{char}}.",
        speech_preferences=session.speech_preferences,
        project_instruction_state=replace(
            session.project_instruction_state,
            project_instruction_notice_key=None,
        ),
    )


def test_fork_registration_failure_publishes_no_partial_indices() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    fence = store.issue_fork_fence(selected.id)
    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id=session.id,
        fork_conversation_id="fork-conversation",
    )
    before_sessions = tuple(item.id for item in store.sessions())
    before_message_index = dict(store._message_session_index)

    with pytest.raises(ValueError, match="session id"):
        store.register_fork_snapshot(snapshot, activate=False)

    assert tuple(item.id for item in store.sessions()) == before_sessions
    assert store._message_session_index == before_message_index


def test_staged_session_and_conversation_id_collision_is_rejected_before_publish() -> (
    None
):
    store, _, session, _, _, _, selected, _ = _fork_store()
    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(selected.id),
        title="Independent fork",
        fork_session_id="same-target-id",
        fork_conversation_id="same-target-id",
    )

    with pytest.raises(ValueError, match="ownership"):
        store.register_fork_snapshot(snapshot, activate=False)

    assert "same-target-id" not in {item.id for item in store.sessions()}
    assert store.active_session_id == session.id


@pytest.mark.parametrize(
    "collision",
    tuple(
        pair
        for pair in combinations(
            ("session", "conversation", "native", "persisted", "turn", "variant"), 2
        )
        if pair != ("native", "persisted")
    ),
)
def test_fork_registration_rejects_cross_domain_ownership_id_collisions(
    collision,
) -> None:
    store = ConsoleChatStore()
    snapshot = _registration_snapshot(collision)

    with pytest.raises(ValueError, match="ownership"):
        store.register_fork_snapshot(snapshot, activate=False)

    assert store.sessions() == []
    assert store._message_session_index == {}


def test_fork_registration_allows_each_messages_native_and_persisted_id_to_match() -> (
    None
):
    store = ConsoleChatStore()
    snapshot = _registration_snapshot()
    snapshot = replace(
        snapshot,
        messages=tuple(
            replace(
                message,
                persisted_message_id=message.native_message_id,
                persisted_parent_id=message.native_parent_id,
            )
            for message in snapshot.messages
        ),
    )

    session = store.register_fork_snapshot(snapshot, activate=False)

    assert all(
        message.id == message.persisted_message_id
        for message in store.messages_for_session(session.id)
    )


def test_fork_registration_rejects_persisted_id_matching_another_message_native_id() -> (
    None
):
    store = ConsoleChatStore()
    snapshot = _registration_snapshot()
    first, second = snapshot.messages
    first = replace(first, persisted_message_id=second.native_message_id)
    second = replace(second, persisted_parent_id=first.persisted_message_id)
    snapshot = replace(snapshot, messages=(first, second))

    with pytest.raises(ValueError, match="ownership"):
        store.register_fork_snapshot(snapshot, activate=False)

    assert store.sessions() == []
    assert store._message_session_index == {}


@pytest.mark.parametrize(
    ("field_name", "invalid_id"),
    (
        ("turn_id", ""),
        ("turn_id", "  "),
        ("visible_variant_id", ""),
        ("visible_variant_id", "  "),
    ),
)
def test_fork_registration_rejects_blank_turn_and_variant_ids(
    field_name,
    invalid_id,
) -> None:
    store = ConsoleChatStore()
    snapshot = _registration_snapshot()
    messages = tuple(
        replace(message, **{field_name: invalid_id})
        if field_name == "turn_id" or message.visible_variant_id is not None
        else message
        for message in snapshot.messages
    )
    snapshot = replace(snapshot, messages=messages)

    with pytest.raises(ValueError, match="ownership"):
        store.register_fork_snapshot(snapshot, activate=False)

    assert store.sessions() == []


def test_fork_registration_allows_one_turn_id_shared_by_multiple_messages() -> None:
    store = ConsoleChatStore()
    snapshot = _registration_snapshot()

    session = store.register_fork_snapshot(snapshot, activate=False)

    assert {message.turn_id for message in store.messages_for_session(session.id)} == {
        "fork-turn"
    }


def test_durable_fork_registration_hydrates_seeded_policy_before_cas(
    tmp_path,
) -> None:
    async def scenario() -> None:
        db = CharactersRAGDB(tmp_path / "fork-policy.sqlite", client_id="fork-policy")
        service = ChatPersistenceService(db)
        snapshot = _registration_snapshot()
        conversation_id = service.create_conversation(
            conversation_id=snapshot.fork_conversation_id,
            conversation_title=snapshot.title,
        )
        assert conversation_id == snapshot.fork_conversation_id
        seeded = service.console_library_policy_repository.insert(
            conversation_id,
            snapshot.configuration.library_policy,
        )
        assert seeded.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
        assert seeded.snapshot.policy_revision == 1

        store = ConsoleChatStore(persistence=service)
        session = store.register_fork_snapshot(snapshot, activate=False)

        assert session.library_policy_hydrated is False
        assert session.library_policy_holder.snapshot.policy_revision is None
        assert session.library_policy_holder.snapshot.source == "missing"
        assert (
            session.library_policy_holder.snapshot.auto_retrieve
            is ConsoleAutoRetrieve.NEVER
        )
        assert (
            session.library_policy_holder.snapshot.assistant_access
            is ConsoleAssistantLibraryAccess.BLOCKED
        )

        hydrated = await store.hydrate_session_library_policy(session.id)
        assert session.library_policy_hydrated is True
        assert hydrated.source == "durable"
        assert hydrated.policy_revision == 1

        edited = ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        )
        store.stage_session_library_policy(session.id, edited)
        saved = await store.save_session_library_policy(session.id)

        assert saved.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
        assert saved.snapshot.policy_revision == 2
        persisted = service.console_library_policy_repository.read(conversation_id)
        assert persisted.snapshot == saved.snapshot

    asyncio.run(scenario())


def test_fork_registration_rolls_back_all_indices_if_activation_raises(
    monkeypatch,
) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(selected.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    before_sessions = tuple(item.id for item in store.sessions())
    before_message_index = dict(store._message_session_index)

    def fail_activation(_session_id: str) -> None:
        raise RuntimeError("activation failed")

    monkeypatch.setattr(store, "_activate_session", fail_activation)

    with pytest.raises(RuntimeError, match="activation failed"):
        store.register_fork_snapshot(snapshot, activate=True)

    assert tuple(item.id for item in store.sessions()) == before_sessions
    assert store._message_session_index == before_message_index
    assert "fork-session" not in store._nodes_by_session
    assert all(
        message.native_message_id not in store._native_parent_by_message
        for message in snapshot.messages
    )
    assert store.active_session_id == session.id
