import pickle
from dataclasses import FrozenInstanceError, fields, replace
from typing import get_args

import pytest

from tldw_chatbook.Chat import console_chat_fork
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
    GenerationVariantMeta,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    sanitize_fork_project_instruction_state,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem


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
        runtime_backend="local",
        assistant_kind="persona",
        assistant_id="persona-1",
        assistant_authority_id="authority-1",
        persona_memory_mode="conversation",
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
        attachments=(
            MessageAttachment(b"sent", "text/plain", "question.txt", 0),
        ),
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
        attachments=(
            MessageAttachment(b"image", "image/png", "selected.png", 0),
        ),
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
            (user.id, first_answer.id, later_user.id, later_answer.id, selected.id, after.id),
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
    return store, persistence, session, user, first_answer, later_answer, selected, after


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
                (node_id, store._message_session_index[node_id])
                for node_id in node_ids
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
        runtime_backend="chat",
        assistant_kind=None,
        assistant_id=None,
        assistant_authority_id=None,
        persona_memory_mode=None,
        character_id=None,
        character_name=None,
        user_display_name_override=None,
        character_system_template=None,
        speech_preferences=ConsoleSpeechPreferences(),
        project_instruction_state=ProjectInstructionControlState.new_session(),
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
    assert fingerprint("test", {"alpha": 1}) != fingerprint(
        "other", {"alpha": 1}
    )


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
    assert all(
        entry.role is not ConsoleMessageRole.TOOL for entry in fence.lineage
    )
    assert store.active_path_message_ids(session.id) == active_before
    assert store.get_message(selected.id).variants.current.id == selected_variant_before


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
        (ConsoleMessageRole.ASSISTANT, "streaming", "partial", False),
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
        selected_live.status = "discarded"
    elif mutation == "parent":
        store._native_parent_by_message[selected.id] = None
    elif mutation == "persisted_parent":
        selected_live.parent_message_id = None
    elif mutation == "selected_variant":
        selected_live.variants.selected_index = 0
        selected_live.content = selected_live.variants.current.content
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
    else:
        persistence.message_versions[selected_live.persisted_message_id] += 1

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
            "runtime_backend": "server",
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
        assert all(message.persisted_message_id is None for message in snapshot.messages)
        assert all(message.persisted_parent_id is None for message in snapshot.messages)
    else:
        assert all(message.persisted_message_id for message in snapshot.messages)
        assert all(
            message.persisted_parent_id is None
            or message.persisted_parent_id
            in {
                projected.persisted_message_id for projected in snapshot.messages
            }
            for message in snapshot.messages
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
        runtime_backend="local",
        assistant_kind="persona",
        assistant_id="persona-1",
        assistant_authority_id="authority-1",
        persona_memory_mode="conversation",
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
