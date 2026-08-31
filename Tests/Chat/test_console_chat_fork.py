import asyncio
import json
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from io import BytesIO
from itertools import combinations
from threading import Event, Thread
from types import SimpleNamespace
from typing import get_args

import pytest
from PIL import Image as PILImage
from pydantic import ValidationError as PydanticValidationError

from tldw_chatbook.Chat import console_chat_fork
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
)
from tldw_chatbook.Chat.console_chat_fork import (
    CONSOLE_FORK_FINGERPRINT_JSON_MAX_BYTES,
    CONSOLE_FORK_TITLE_MAX_LENGTH,
    CONSOLE_FORK_VIDEO_TOMBSTONE_CONTENT,
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
    ConsoleForkProjectedVideoTombstone,
    default_fork_title,
    encode_console_fork_message_metadata,
    normalize_fork_title,
    parse_console_fork_message_metadata,
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
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Video_Generation.video_store import video_content_marker
from tldw_chatbook.Utils import input_validation


class _ForkVersionPersistence:
    db = None
    # Fixture messages are intentionally process-local until this helper
    # assigns synthetic durable ids below. This explicit capability keeps the
    # production adapter guard fail-closed for genuinely persisted owners.
    console_process_local_only = True

    def __init__(self) -> None:
        self.conversation_version = 7
        self.active_leaf_by_conversation: dict[str, str | None] = {}
        self.message_versions: dict[str, int] = {}
        self.message_bodies: dict[str, str] = {}
        self.citation_states: dict[str, str] = {}

    def get_conversation_version(self, _conversation_id: str) -> int:
        return self.conversation_version

    def get_console_fork_active_leaf(self, conversation_id: str) -> str | None:
        return self.active_leaf_by_conversation.get(conversation_id)

    def get_message_version(self, message_id: str) -> int | None:
        return self.message_versions.get(message_id)

    def get_console_fork_source_message(
        self, message_id: str
    ) -> tuple[int, str] | None:
        version = self.message_versions.get(message_id)
        body = self.message_bodies.get(message_id)
        return (version, body) if version is not None and body is not None else None

    def get_console_fork_citation_state(
        self,
        message_id: str,
        revision: int,
        source_body: str,
        target_body: str,
    ) -> tuple[str, str | None]:
        assert revision == self.message_versions[message_id]
        assert source_body == self.message_bodies[message_id]
        assert target_body
        state = self.citation_states.get(message_id, "none")
        if state == "ambiguous":
            raise CitationPersistenceUnavailable("fork_owner_ambiguous")
        if state == "active_required" and source_body != target_body:
            return "unavailable", None
        return state, ("trace-1" if state == "active_required" else None)


def test_fork_message_metadata_reads_v1_and_round_trips_v2_trace_turn_id() -> None:
    legacy = json.dumps(
        {
            "console_fork": {
                "version": 1,
                "status": "stopped",
                "attachment_display_name": "legacy.png",
            }
        }
    )

    assert parse_console_fork_message_metadata(legacy) == (
        "stopped",
        "legacy.png",
        None,
    )
    encoded = encode_console_fork_message_metadata(
        "complete",
        "",
        "source-trace-turn",
    )
    assert parse_console_fork_message_metadata(encoded) == (
        "complete",
        "",
        "source-trace-turn",
    )


def _image_bytes(
    color: tuple[int, int, int] = (0, 0, 0),
    *,
    image_format: str = "PNG",
) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", (2, 2), color).save(buffer, format=image_format)
    return buffer.getvalue()


def _fork_store(
    *,
    durable: bool = False,
    ephemeral: bool = False,
    generated_image: bool = False,
):
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
        attachments=(
            MessageAttachment(_image_bytes(), "image/png", "selected.png", 0),
        ),
    )
    store.add_variant(selected.id, "Selected variant")
    selected_live = store._nodes_by_session[session.id][selected.id]
    if generated_image:
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
            persistence.message_bodies[message.persisted_message_id] = message.content
        for message_id in store._nodes_by_session[session.id]:
            parent_id = store._native_parent_by_message[message_id]
            store._nodes_by_session[session.id][message_id].parent_message_id = (
                store._nodes_by_session[session.id][parent_id].persisted_message_id
                if parent_id is not None
                else None
            )
        active_leaf = store._active_leaf_by_session[session.id]
        persistence.active_leaf_by_conversation["conversation-1"] = (
            store._nodes_by_session[session.id][active_leaf].persisted_message_id
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


def test_fork_configuration_fingerprint_includes_normalized_thinking_policy() -> None:
    configuration = _configuration_snapshot()

    included = replace(configuration, thinking_history_policy="include")
    excluded = replace(configuration, thinking_history_policy="exclude")

    assert console_chat_fork.fingerprint_console_fork_configuration(included) != (
        console_chat_fork.fingerprint_console_fork_configuration(excluded)
    )
    with pytest.raises((TypeError, ValueError), match="thinking"):
        console_chat_fork.fingerprint_console_fork_configuration(
            replace(configuration, thinking_history_policy="required")
        )


def _image_selection(
    message,
    *,
    position: int,
    browse_revision: int = 0,
) -> ConsoleForkImageSelectionFence:
    return ConsoleForkImageSelectionFence(
        native_message_id=message.id,
        selected_position=position,
        browse_revision=browse_revision,
        attachment_meta_fingerprint=(
            console_chat_fork.fingerprint_console_fork_selected_image(
                message.attachments[position],
                message.generation_metadata[position],
            )
        ),
    )


def _generation_meta_for_video() -> GenerationVariantMeta:
    return GenerationVariantMeta(
        prompt="source image",
        negative_prompt="",
        backend="openai",
        model="image-test",
        seed=7,
        style=None,
        params={"size": "small"},
    )


def _new_fork_session(store: ConsoleChatStore, *, title: str):
    return store.create_session(
        title=title,
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
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
            source_persisted_content=None,
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
            source_persisted_content=None,
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
        source_conversation_version=None,
        source_active_leaf_persisted_message_id=None,
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


def test_fork_title_uses_the_shared_strict_validation_boundary() -> None:
    assert hasattr(input_validation, "ConsoleForkTitleInput")
    model_cls = getattr(input_validation, "ConsoleForkTitleInput")

    assert model_cls.model_validate({"title": "  proposed\n title  "}).title == (
        "proposed title"
    )
    with pytest.raises(PydanticValidationError):
        model_cls.model_validate({"title": " \n\t "})
    with pytest.raises(PydanticValidationError):
        model_cls.model_validate({"title": 123})


def test_fork_title_reuses_the_console_title_deriver(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_derive(draft: str, *, max_length: int) -> str:
        calls.append((draft, max_length))
        return "canonical title"

    monkeypatch.setattr(
        input_validation,
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
        ConsoleForkProjectedVideoTombstone,
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
                "persisted_content",
                "attachment_fingerprint",
                "trace_turn_id",
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
                "source_active_leaf_persisted_message_id",
                "source_durability",
                "source_title",
                "source_configuration_fingerprint",
                "boundary_message_id",
                    "lineage",
                    "image_selections",
                    "trace_boundary",
            ),
        ),
        (
            ConsoleForkProjectedMessage,
            (
                "source_native_message_id",
                "source_persisted_message_id",
                "source_persisted_revision",
                "source_persisted_content",
                "native_message_id",
                "persisted_message_id",
                "native_parent_id",
                "persisted_parent_id",
                "turn_id",
                "visible_variant_id",
                "role",
                "status",
                "content",
                "trace_turn_id",
                "attachments",
                "generation_metadata",
                "video_tombstone",
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
            ConsoleForkProjectedVideoTombstone,
            (
                "owner_native_message_id",
                "owner_persisted_message_id",
                "source_fingerprint",
                "prompt",
                "negative_prompt",
                "backend",
                "model",
                "seed",
                "duration_seconds",
                "fps",
                "width",
                "height",
                "ratio",
                "source_image_message_id",
                "container",
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
                "thinking_history_policy",
            ),
        ),
        (
            ConsoleForkCitationLink,
            (
                "source_persisted_message_id",
                "source_revision",
                "state",
                "trace_id",
            ),
        ),
        (
            ConsoleChatForkSnapshot,
            (
                "fork_session_id",
                "fork_conversation_id",
                "title",
                "source_session_id",
                "source_conversation_id",
                "source_conversation_version",
                "source_active_leaf_persisted_message_id",
                "source_boundary_persisted_message_id",
                "durable",
                "messages",
                    "configuration",
                    "citation_links",
                    "trace_boundary",
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


def test_durable_fork_fences_saved_database_leaf_before_unsaved_live_tail() -> None:
    store, persistence, session, _, _, _, selected, after = _fork_store(durable=True)
    selected_persisted_id = store.get_message(selected.id).persisted_message_id
    assert selected_persisted_id is not None
    persistence.active_leaf_by_conversation["conversation-1"] = selected_persisted_id
    unsaved_tail = store._nodes_by_session[session.id][after.id]
    unsaved_tail.persisted_message_id = None

    eligibility = store.fork_eligibility(selected.id)
    fence = store.issue_fork_fence(selected.id)
    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )

    assert eligibility.eligible is True
    assert fence.source_active_leaf_persisted_message_id == selected_persisted_id
    assert snapshot.source_active_leaf_persisted_message_id == selected_persisted_id
    assert after.id not in {
        message.source_native_message_id for message in snapshot.messages
    }


def test_durable_fork_rejects_boundary_outside_database_active_lineage() -> None:
    store, persistence, _, _, _, later_answer, selected, _ = _fork_store(durable=True)
    persistence.active_leaf_by_conversation["conversation-1"] = store.get_message(
        later_answer.id
    ).persisted_message_id

    with pytest.raises(ValueError, match="active leaf"):
        store.issue_fork_fence(selected.id)


def test_issue_fork_fence_captures_the_exact_image_selection_tuple() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store(generated_image=True)
    selection = _image_selection(
        store._nodes_by_session[session.id][selected.id],
        position=0,
        browse_revision=7,
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
    store, _, session, _, _, _, selected, _ = _fork_store(generated_image=True)
    selection = _image_selection(
        store._nodes_by_session[session.id][selected.id],
        position=0,
        browse_revision=7,
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


def test_quarantined_active_prefix_blocks_fork_fence_and_staging() -> None:
    store, _, session, user, _, _, selected, _ = _fork_store()
    fence = store.issue_fork_fence(selected.id)
    source_session_ids = tuple(item.id for item in store.sessions())
    source_message_ids = tuple(store._message_session_index)
    quarantined = store._nodes_by_session[session.id][user.id]
    store._quarantine_generation_projection(
        quarantined,
        minimum_version=2,
        reason="Canonical generation is unavailable; reload required.",
    )

    eligibility = store.fork_eligibility(selected.id)
    assert not eligibility.eligible
    assert "reload" in eligibility.reason.lower()
    with pytest.raises(ValueError, match="reload"):
        store.issue_fork_fence(selected.id)
    assert not store.validate_fork_fence(fence)
    with pytest.raises(ValueError, match="source changed"):
        store.stage_fork_snapshot(
            fence,
            title="Rejected fork",
            fork_session_id="fork-rejected",
            fork_conversation_id=None,
        )
    assert tuple(item.id for item in store.sessions()) == source_session_ids
    assert tuple(store._message_session_index) == source_message_ids


def test_quarantine_cannot_interleave_after_fork_validation(monkeypatch) -> None:
    store, _, session, user, _, _, selected, _ = _fork_store()
    fence = store.issue_fork_fence(selected.id)
    source = store._nodes_by_session[session.id][user.id]
    validated = Event()
    release_stage = Event()
    quarantine_finished = Event()
    snapshots: list[ConsoleChatForkSnapshot] = []
    failures: list[BaseException] = []
    original_validate = store._validate_fork_fence

    def blocking_validate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        validated.set()
        assert release_stage.wait(2)
        return result

    monkeypatch.setattr(store, "_validate_fork_fence", blocking_validate)

    def stage() -> None:
        try:
            snapshots.append(
                store.stage_fork_snapshot(
                    fence,
                    title="Atomic fork",
                    fork_session_id="fork-atomic",
                    fork_conversation_id="fork-conversation",
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            failures.append(exc)

    def quarantine() -> None:
        store._quarantine_generation_projection(
            source,
            minimum_version=2,
            reason="Canonical generation is unavailable; reload required.",
        )
        quarantine_finished.set()

    stage_thread = Thread(target=stage)
    quarantine_thread = Thread(target=quarantine)
    stage_thread.start()
    assert validated.wait(2)
    quarantine_thread.start()
    assert not quarantine_finished.wait(0.1)
    release_stage.set()
    stage_thread.join(2)
    quarantine_thread.join(2)

    assert not failures
    assert len(snapshots) == 1
    assert quarantine_finished.is_set()
    assert source.generation_projection_quarantined


def test_content_edit_cannot_interleave_after_fork_validation(monkeypatch) -> None:
    store, _, _, _, _, _, selected, _ = _fork_store()
    fence = store.issue_fork_fence(selected.id)
    validated = Event()
    release_stage = Event()
    edit_finished = Event()
    snapshots: list[ConsoleChatForkSnapshot] = []
    failures: list[BaseException] = []
    original_validate = store._validate_fork_fence

    def blocking_validate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        validated.set()
        assert release_stage.wait(2)
        return result

    monkeypatch.setattr(store, "_validate_fork_fence", blocking_validate)

    def stage() -> None:
        try:
            snapshots.append(
                store.stage_fork_snapshot(
                    fence,
                    title="Atomic fork",
                    fork_session_id="fork-atomic-edit",
                    fork_conversation_id="fork-conversation",
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            failures.append(exc)

    def edit() -> None:
        try:
            store.update_message_content(selected.id, "Edited after fork")
        except BaseException as exc:  # pragma: no cover - assertion reports it
            failures.append(exc)
        finally:
            edit_finished.set()

    stage_thread = Thread(target=stage)
    edit_thread = Thread(target=edit)
    stage_thread.start()
    assert validated.wait(2)
    edit_thread.start()
    assert not edit_finished.wait(0.1)
    release_stage.set()
    stage_thread.join(2)
    edit_thread.join(2)

    assert not failures
    assert len(snapshots) == 1
    assert edit_finished.is_set()
    projected = next(
        item for item in snapshots[0].messages if item.content == "Selected variant"
    )
    assert projected.content == "Selected variant"
    assert store.get_message(selected.id).content == "Edited after fork"


def test_generation_commit_blocks_fork_until_live_owner_is_published(
    tmp_path, monkeypatch
) -> None:
    db = CharactersRAGDB(
        tmp_path / "fork-generation-transition.sqlite",
        client_id="fork-generation-transition",
    )
    try:
        service = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=service)
        session = store.create_session(
            settings=ConsoleSessionSettings(provider="openai", model="gpt-test")
        )
        session.library_policy_hydrated = True
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Question",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Original",
            persist=True,
        )
        committed = Event()
        release_writer = Event()
        writer_failures: list[BaseException] = []
        original_writer = service.replace_assistant_generation_projection

        def blocking_writer(**kwargs):
            result = original_writer(**kwargs)
            committed.set()
            assert release_writer.wait(2)
            return result

        monkeypatch.setattr(
            service, "replace_assistant_generation_projection", blocking_writer
        )

        def add_variant() -> None:
            try:
                store.add_variant(assistant.id, "Durable replacement")
            except BaseException as exc:  # pragma: no cover - assertion reports it
                writer_failures.append(exc)

        writer_thread = Thread(target=add_variant)
        writer_thread.start()
        assert committed.wait(2)

        eligibility = store.fork_eligibility(assistant.id)
        assert not eligibility.eligible
        assert "changing" in eligibility.reason.lower()
        with pytest.raises(ValueError, match="changing"):
            store.issue_fork_fence(assistant.id)

        release_writer.set()
        writer_thread.join(2)
        assert not writer_thread.is_alive()
        assert writer_failures == []
        assert store.get_message(assistant.id).content == "Durable replacement"
        assert db.get_message_by_id(assistant.persisted_message_id)["content"] == (
            "Durable replacement"
        )
    finally:
        db.close_connection()


def test_nested_fork_source_transitions_balance_and_do_not_block_other_sessions() -> (
    None
):
    store, _, session, _, _, _, selected, _ = _fork_store()
    other_store, _, other_session, _, _, _, other_selected, _ = _fork_store()
    # Use the same store so the assertion covers per-session isolation.
    other = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test")
    )
    other_message = store.append_message(
        other.id,
        role=ConsoleMessageRole.USER,
        content="Independent question",
    )

    with store.fork_source_transition(session.id):
        assert store._fork_source_transitions[session.id] == 1
        with store.fork_source_transition(session.id):
            assert store._fork_source_transitions[session.id] == 2
            assert store.fork_eligibility(selected.id).eligible is False
            assert store.fork_eligibility(other_message.id).eligible is True
        assert store._fork_source_transitions[session.id] == 1
    assert session.id not in store._fork_source_transitions
    assert store.fork_eligibility(selected.id).eligible is True
    assert other_store.fork_eligibility(other_selected.id).eligible is True
    assert other_session.id not in other_store._fork_source_transitions


def test_detached_roleplay_plan_blocks_fork_until_owner_accepts_result() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store(durable=True)
    live_session = store._sessions[session.id]
    live_session.assistant_kind = "character"
    live_session.character_name = "Nova"
    live_session.persona_memory_mode = None
    live_session.character_system_template = "You are {{char}} helping {{user}}."
    live_session.settings = replace(live_session.settings, system_prompt="stale")

    plan = store.prepare_session_roleplay_projection_refresh(
        session.id,
        global_default="Riley",
        force_persistence=True,
    )

    assert plan is not None
    assert plan.fork_transition_token is not None
    assert store._roleplay_fork_transition_leases == {
        plan.fork_transition_token: session.id
    }
    assert store.fork_eligibility(selected.id).eligible is False
    result = store.persist_roleplay_projection_plan(plan)
    assert store.fork_eligibility(selected.id).eligible is False
    assert store.accept_roleplay_projection_persistence_result(result) is True
    assert store.fork_eligibility(selected.id).eligible is True
    assert session.id not in store._fork_source_transitions
    assert store._roleplay_fork_transition_leases == {}
    assert (
        store.prepare_session_roleplay_projection_refresh(
            session.id,
            global_default="Riley",
        )
        is None
    )
    assert store._fork_source_transitions == {}
    assert store._roleplay_fork_transition_leases == {}


def test_abandoned_roleplay_plan_releases_only_its_transition_lease() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store(durable=True)
    live_session = store._sessions[session.id]
    live_session.assistant_kind = "character"
    live_session.character_name = "Nova"
    live_session.persona_memory_mode = None
    live_session.character_system_template = "You are {{char}} helping {{user}}."
    live_session.settings = replace(live_session.settings, system_prompt="stale")
    plan = store.prepare_session_roleplay_projection_refresh(
        session.id,
        global_default="Riley",
        force_persistence=True,
    )
    assert plan is not None

    with store.fork_source_transition(session.id):
        assert store.abandon_roleplay_projection_plan(plan) is True
        assert store._fork_source_transitions[session.id] == 1
        assert store.fork_eligibility(selected.id).eligible is False
    assert store.fork_eligibility(selected.id).eligible is True
    assert store.abandon_roleplay_projection_plan(plan) is False
    assert store._fork_source_transitions == {}
    assert store._roleplay_fork_transition_leases == {}


@pytest.mark.parametrize("outcome", ("none", "exception"))
def test_roleplay_materialization_failure_releases_transition_lease(
    monkeypatch,
    outcome: str,
) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store(durable=True)
    live_session = store._sessions[session.id]
    live_session.assistant_kind = "character"
    live_session.character_name = "Nova"
    live_session.persona_memory_mode = None
    live_session.character_system_template = "You are {{char}} helping {{user}}."

    def materialize(*_args, **_kwargs):
        if outcome == "exception":
            raise RuntimeError("materialization failed")
        return None

    monkeypatch.setattr(store, "_materialize_roleplay_projections_live", materialize)

    if outcome == "exception":
        with pytest.raises(RuntimeError, match="materialization failed"):
            store.prepare_session_roleplay_projection_refresh(
                session.id,
                global_default="Riley",
                force_persistence=True,
            )
    else:
        assert (
            store.prepare_session_roleplay_projection_refresh(
                session.id,
                global_default="Riley",
                force_persistence=True,
            )
            is None
        )

    assert store._fork_source_transitions == {}
    assert store._roleplay_fork_transition_leases == {}
    assert store.fork_eligibility(selected.id).eligible is True


@pytest.mark.parametrize("route", ("active_leaf", "system_prompt"))
def test_configuration_and_leaf_writers_block_fork_through_live_publication(
    monkeypatch,
    route,
) -> None:
    store, persistence, session, _, first_answer, _, selected, _ = _fork_store(
        durable=True
    )
    entered = Event()
    release = Event()
    failures: list[BaseException] = []

    if route == "active_leaf":

        def blocking_leaf(_session_id, _message_id):
            entered.set()
            assert release.wait(2)

        monkeypatch.setattr(store, "_persist_active_leaf", blocking_leaf)

        def mutate() -> None:
            store.set_active_leaf(session.id, first_answer.id)
    else:

        def blocking_prompt(**_kwargs):
            entered.set()
            assert release.wait(2)
            return True

        monkeypatch.setattr(
            persistence,
            "update_conversation_system_prompt",
            blocking_prompt,
            raising=False,
        )

        def mutate() -> None:
            store.set_session_system_prompt(session.id, "Changed system prompt")

    def run_mutation() -> None:
        try:
            mutate()
        except BaseException as exc:  # pragma: no cover - assertion reports it
            failures.append(exc)

    thread = Thread(target=run_mutation)
    thread.start()
    assert entered.wait(2)
    assert store.fork_eligibility(selected.id).eligible is False
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    assert failures == []
    assert session.id not in store._fork_source_transitions


def test_image_recovery_merge_blocks_fork_while_canonical_row_is_hydrated(
    tmp_path,
    monkeypatch,
) -> None:
    db = CharactersRAGDB(tmp_path / "fork-image-recovery.sqlite", "fork-image")
    try:
        service = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=service)
        session = store.create_session(
            title="Image recovery",
            settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
        )
        session.library_policy_hydrated = True
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Draw it",
            persist=True,
        )
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None
        recovered_id = "recovered-image-message"
        service.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            content="Recovered image",
            message_id=recovered_id,
            parent_message_id=user.persisted_message_id,
            attachments=[
                {
                    "position": 0,
                    "data": _image_bytes(),
                    "mime_type": "image/png",
                }
            ],
            generation_metadata=[
                {
                    "position": 0,
                    "prompt": "recovered",
                    "backend": "openai",
                }
            ],
        )
        entered = Event()
        release = Event()
        failures: list[BaseException] = []
        original = db.get_message_by_id

        def blocking_read(message_id):
            row = original(message_id)
            if message_id == recovered_id:
                entered.set()
                assert release.wait(2)
            return row

        monkeypatch.setattr(db, "get_message_by_id", blocking_read)

        def merge() -> None:
            try:
                store.merge_persisted_generation_message(session.id, recovered_id)
            except BaseException as exc:  # pragma: no cover - assertion reports it
                failures.append(exc)

        thread = Thread(target=merge)
        thread.start()
        assert entered.wait(2)
        eligibility = store.fork_eligibility(user.id)
        assert eligibility.eligible is False
        assert "changing" in eligibility.reason.lower()
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert failures == []
        recovered = store.get_message(recovered_id)
        assert recovered.content == "Recovered image"
        assert recovered.attachments[0].data == _image_bytes()
        assert session.id not in store._fork_source_transitions
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    ("private_route", "invoke", "result_factory"),
    (
        (
            "_delete_message",
            lambda store, session, message: store.delete_message(message.id),
            lambda store, session, message: store._snapshot(message),
        ),
        (
            "_append_generation_variant",
            lambda store, session, message: store.append_generation_variant(
                session.id,
                message.id,
                data=b"variant",
                mime_type="image/png",
                meta=GenerationVariantMeta("prompt", "", "local", None, None, None, {}),
            ),
            lambda store, session, message: 1,
        ),
        (
            "_keep_generation_variant",
            lambda store, session, message: store.keep_generation_variant(
                session.id, message.id, position=1
            ),
            lambda store, session, message: None,
        ),
        (
            "_set_message_metadata",
            lambda store, session, message: store.set_message_metadata(
                message.id, MessageMetadata()
            ),
            lambda store, session, message: store._snapshot(message),
        ),
        (
            "_set_message_usage",
            lambda store, session, message: store.set_message_usage(
                message.id, ProviderUsage(uncached_input=1, output=1)
            ),
            lambda store, session, message: store._snapshot(message),
        ),
        (
            "_set_session_character_name",
            lambda store, session, message: store.set_session_character_name(
                session.id, "Changed", global_default="User"
            ),
            lambda store, session, message: (session, True),
        ),
    ),
)
def test_fork_rejects_each_source_transition_route_class(
    monkeypatch, private_route, invoke, result_factory
) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store()
    entered = Event()
    release = Event()
    finished = Event()
    failures: list[BaseException] = []

    def blocking_route(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return result_factory(store, session, selected)

    monkeypatch.setattr(store, private_route, blocking_route)

    def mutate() -> None:
        try:
            invoke(store, session, selected)
        except BaseException as exc:  # pragma: no cover - assertion reports it
            failures.append(exc)
        finally:
            finished.set()

    mutation_thread = Thread(target=mutate)
    mutation_thread.start()
    assert entered.wait(2)

    eligibility = store.fork_eligibility(selected.id)
    assert not eligibility.eligible
    assert "changing" in eligibility.reason.lower()

    release.set()
    mutation_thread.join(2)
    assert not mutation_thread.is_alive()
    assert finished.is_set()
    assert failures == []


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
        durable=True,
        generated_image=True,
    )
    selected_live = store._nodes_by_session[session.id][selected.id]
    selection = _image_selection(selected_live, position=0)
    fence = store.issue_fork_fence(
        selected.id,
        image_selections=(selection,),
    )

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
            MessageAttachment(
                _image_bytes((255, 0, 0)),
                "image/png",
                "selected.png",
                0,
            ),
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

    if mutation in {"status", "turn", "selected_variant"}:
        assert store.fork_eligibility(selected.id).eligible is True
    assert store.validate_fork_fence(fence, image_selections=(selection,)) is False


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


@pytest.mark.parametrize("first_state", ("none", "unavailable"))
def test_stage_fork_snapshot_freezes_each_durable_citation_state(
    first_state: str,
) -> None:
    store, persistence, session, user, _, _, selected, _ = _fork_store(durable=True)
    source_user = store._nodes_by_session[session.id][user.id]
    source_selected = store._nodes_by_session[session.id][selected.id]
    persistence.citation_states[source_user.persisted_message_id] = first_state
    persistence.citation_states[source_selected.persisted_message_id] = (
        "active_required"
    )
    fence = store.issue_fork_fence(selected.id)

    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )

    assert snapshot.citation_links == (
        ConsoleForkCitationLink(
            source_persisted_message_id=source_user.persisted_message_id,
            source_revision=persistence.message_versions[
                source_user.persisted_message_id
            ],
            state=first_state,
            trace_id=None,
        ),
        ConsoleForkCitationLink(
            source_persisted_message_id=source_selected.persisted_message_id,
            source_revision=persistence.message_versions[
                source_selected.persisted_message_id
            ],
            state="active_required",
            trace_id="trace-1",
        ),
    )


def test_stage_fork_marks_active_provenance_unavailable_for_visible_variant() -> None:
    store, persistence, session, _, _, _, selected, _ = _fork_store(durable=True)
    source = store._nodes_by_session[session.id][selected.id]
    source_id = source.persisted_message_id
    assert source_id is not None
    persistence.message_bodies[source_id] = "Canonical governed source [S1]."
    persistence.citation_states[source_id] = "active_required"

    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(selected.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected = snapshot.messages[-1]

    assert projected.source_persisted_content == "Canonical governed source [S1]."
    assert projected.content != projected.source_persisted_content
    assert snapshot.citation_links[-1].state == "unavailable"


def test_stage_fork_marks_active_video_provenance_unavailable_for_tombstone() -> None:
    persistence = _ForkVersionPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = _new_fork_session(store, title="Video source")
    session.persisted_conversation_id = "conversation-1"
    user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Animate this",
    )
    video = store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video-key",
            prompt="animate",
            backend="minimax",
        ),
    )
    for revision, message in enumerate((user, video), start=1):
        live = store._nodes_by_session[session.id][message.id]
        live.persisted_message_id = f"persisted-{revision}"
        live.parent_message_id = None if revision == 1 else "persisted-1"
        persistence.message_versions[live.persisted_message_id] = revision
        persistence.message_bodies[live.persisted_message_id] = live.content
    video_source_id = store._nodes_by_session[session.id][video.id].persisted_message_id
    assert video_source_id is not None
    persistence.active_leaf_by_conversation["conversation-1"] = video_source_id
    persistence.citation_states[video_source_id] = "active_required"

    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(video.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected = snapshot.messages[-1]

    assert projected.source_persisted_content == video_content_marker(
        "source-video-key"
    )
    assert projected.content != projected.source_persisted_content
    assert projected.video_tombstone is not None
    assert snapshot.citation_links[-1].state == "unavailable"


def test_stage_fork_snapshot_rejects_ambiguous_citation_authority() -> None:
    store, persistence, session, _, _, _, selected, _ = _fork_store(durable=True)
    source_selected = store._nodes_by_session[session.id][selected.id]
    persistence.citation_states[source_selected.persisted_message_id] = "ambiguous"
    fence = store.issue_fork_fence(selected.id)

    with pytest.raises(ValueError, match="citation authority"):
        store.stage_fork_snapshot(
            fence,
            title="Independent fork",
            fork_session_id="fork-session",
            fork_conversation_id="fork-conversation",
        )


def test_temporary_fork_snapshot_keeps_text_without_governed_citation_links() -> None:
    store, _, _, _, _, _, selected, _ = _fork_store(ephemeral=True)
    fence = store.issue_fork_fence(selected.id)

    snapshot = store.stage_fork_snapshot(
        fence,
        title="Temporary fork",
        fork_session_id="fork-session",
        fork_conversation_id=None,
    )

    assert "Selected variant" in [message.content for message in snapshot.messages]
    assert snapshot.citation_links == ()


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
    store, _, session, _, _, _, selected, _ = _fork_store(generated_image=True)
    selection = _image_selection(
        store._nodes_by_session[session.id][selected.id],
        position=0,
    )
    fence = store.issue_fork_fence(
        selected.id,
        image_selections=(selection,),
    )
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
                replace(
                    selected_live.attachments[0],
                    data=_image_bytes((255, 0, 0)),
                ),
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


@pytest.mark.parametrize("mutation", ("attachment", "generation"))
def test_stage_fork_snapshot_rejects_aba_payload_mutation(
    mutation,
    monkeypatch,
) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store(generated_image=True)
    selection = _image_selection(
        store._nodes_by_session[session.id][selected.id],
        position=0,
    )
    fence = store.issue_fork_fence(
        selected.id,
        image_selections=(selection,),
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
                    replace(
                        original_attachments[0],
                        data=_image_bytes((255, 0, 0)),
                    ),
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


def test_stage_fork_snapshot_rebuilds_only_selected_generated_image() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store(generated_image=True)
    selected_live = store._nodes_by_session[session.id][selected.id]
    selected_live.attachments = (
        *selected_live.attachments,
        MessageAttachment(
            _image_bytes((0, 255, 0)),
            "image/png",
            "second.png",
            1,
        ),
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
    selection = _image_selection(selected_live, position=1, browse_revision=1)
    fence = store.issue_fork_fence(
        selected.id,
        image_selections=(selection,),
    )

    assert fence.image_selections == (selection,)
    assert store.validate_fork_fence(fence, image_selections=(selection,)) is True

    snapshot = store.stage_fork_snapshot(
        fence,
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected = snapshot.messages[-1]

    assert len(projected.attachments) == 1
    assert projected.attachments[0].data == _image_bytes((0, 255, 0))
    assert projected.attachments[0].position == 0
    assert (
        projected.attachments[0].owner_native_message_id == projected.native_message_id
    )
    assert len(projected.generation_metadata) == 1
    assert projected.generation_metadata[0].prompt == "second prompt"
    assert projected.generation_metadata[0].position == 0
    assert (
        projected.generation_metadata[0].owner_native_message_id
        == projected.native_message_id
    )


@pytest.mark.parametrize("invalid_kind", ("corrupt", "truncated", "mime-mismatch"))
def test_fork_image_payload_validation_rejects_noncanonical_bytes(
    invalid_kind,
) -> None:
    payload = _image_bytes()
    mime_type = "image/png"
    if invalid_kind == "corrupt":
        payload = b"not an image"
    elif invalid_kind == "truncated":
        payload = payload[:-12]
    else:
        mime_type = "image/jpeg"

    with pytest.raises(ValueError, match="image"):
        console_chat_fork.validate_console_fork_image_payload(payload, mime_type)


@pytest.mark.parametrize(
    ("payload", "mime_type"),
    (
        (b"not an image", "image/png"),
        (_image_bytes(), "image/jpeg"),
    ),
)
def test_fork_fence_rejects_a_noncanonical_declared_image(
    payload,
    mime_type,
) -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Source")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Attached image",
        attachments=(MessageAttachment(payload, mime_type, "source.png", 0),),
    )

    with pytest.raises(ValueError, match="image"):
        store.issue_fork_fence(message.id)


def test_fork_registration_rejects_a_forged_corrupt_image_without_publication() -> None:
    snapshot = _registration_snapshot()
    first, *rest = snapshot.messages
    forged_attachment = ConsoleForkProjectedAttachment(
        owner_native_message_id=first.native_message_id,
        owner_persisted_message_id=first.persisted_message_id,
        position=0,
        data=b"not an image",
        mime_type="image/png",
        display_name="forged.png",
    )
    snapshot = replace(
        snapshot,
        messages=(replace(first, attachments=(forged_attachment,)), *rest),
    )
    store = ConsoleChatStore()

    with pytest.raises(ValueError, match="image"):
        store.register_fork_snapshot(snapshot, activate=False)

    assert store.sessions() == []


def test_fork_registration_rejects_generated_metadata_for_a_non_image() -> None:
    snapshot = _registration_snapshot()
    first, generated = snapshot.messages
    attachment = ConsoleForkProjectedAttachment(
        owner_native_message_id=generated.native_message_id,
        owner_persisted_message_id=generated.persisted_message_id,
        position=0,
        data=b"not an image",
        mime_type="text/plain",
        display_name="forged.txt",
    )
    metadata = ConsoleForkProjectedGeneration(
        owner_native_message_id=generated.native_message_id,
        owner_persisted_message_id=generated.persisted_message_id,
        position=0,
        prompt="forged",
        negative_prompt="",
        backend="openai",
        model=None,
        seed=None,
        style=None,
        params_json="{}",
    )
    snapshot = replace(
        snapshot,
        messages=(
            first,
            replace(
                generated,
                attachments=(attachment,),
                generation_metadata=(metadata,),
            ),
        ),
    )
    store = ConsoleChatStore()

    with pytest.raises(ValueError, match="image"):
        store.register_fork_snapshot(snapshot, activate=False)

    assert store.sessions() == []


def test_fork_fence_requires_a_selection_for_every_generated_image() -> None:
    store, _, session, _, _, _, selected, _ = _fork_store(generated_image=True)
    selected_live = store._nodes_by_session[session.id][selected.id]
    second = store.append_generation_message(
        session.id,
        content="[image] second",
        variants=[
            (_image_bytes((0, 255, 0)), "image/png", _generation_meta_for_video())
        ],
    )

    with pytest.raises(ValueError, match="image selection"):
        store.issue_fork_fence(
            second.id,
            image_selections=(_image_selection(selected_live, position=0),),
        )


def test_generated_image_fork_fails_closed_without_a_selection() -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Source")
    generated = store.append_generation_message(
        session.id,
        content="[image] generated",
        variants=[(_image_bytes(), "image/png", _generation_meta_for_video())],
    )

    with pytest.raises(ValueError, match="image selection"):
        store.issue_fork_fence(generated.id)


def test_stage_fork_snapshot_rejects_a_generated_fence_with_no_selection() -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Source")
    generated = store.append_generation_message(
        session.id,
        content="[image] generated",
        variants=[(_image_bytes(), "image/png", _generation_meta_for_video())],
    )
    fence = store.issue_fork_fence(
        generated.id,
        image_selections=(_image_selection(generated, position=0),),
    )

    with pytest.raises(ValueError, match="source changed"):
        store.stage_fork_snapshot(
            replace(fence, image_selections=()),
            title="Independent fork",
            fork_session_id="fork-session",
            fork_conversation_id="fork-conversation",
        )


@pytest.mark.parametrize("mime_type", ("text/plain", "application/octet-stream"))
def test_generated_image_fork_rejects_non_image_attachment_data(mime_type) -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Source")
    generated = store.append_generation_message(
        session.id,
        content="[image] malformed",
        variants=[(b"not-an-image", mime_type, _generation_meta_for_video())],
    )
    forged_selection = ConsoleForkImageSelectionFence(
        native_message_id=generated.id,
        selected_position=0,
        browse_revision=0,
        attachment_meta_fingerprint="0" * 64,
    )

    with pytest.raises(ValueError, match="image selection"):
        store.issue_fork_fence(
            generated.id,
            image_selections=(forged_selection,),
        )


def test_generated_image_fork_rejects_a_selection_for_a_user_role() -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Source")
    generated = store.append_generation_message(
        session.id,
        content="[image] generated",
        variants=[(_image_bytes(), "image/png", _generation_meta_for_video())],
    )
    selection = _image_selection(generated, position=0)
    store._nodes_by_session[session.id][generated.id].role = ConsoleMessageRole.USER

    with pytest.raises(ValueError, match="image selection"):
        store.issue_fork_fence(
            generated.id,
            image_selections=(selection,),
        )


@pytest.mark.parametrize("invalid_kind", ("duplicate", "off-prefix"))
def test_generated_image_fork_rejects_non_exact_selection_sets(invalid_kind) -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Source")
    off_prefix = store.append_generation_message(
        session.id,
        content="[image] old branch",
        variants=[
            (_image_bytes((255, 0, 0)), "image/png", _generation_meta_for_video())
        ],
    )
    active = store.create_sibling(
        off_prefix.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="[image] active branch",
        attachments=(
            MessageAttachment(
                _image_bytes((0, 255, 0)),
                "image/png",
                "active.png",
                0,
            ),
        ),
    )
    active_live = store._nodes_by_session[session.id][active.id]
    active_live.generation_metadata = (_generation_meta_for_video(),)
    active_selection = _image_selection(active_live, position=0)
    invalid_selections = (
        (active_selection, active_selection)
        if invalid_kind == "duplicate"
        else (active_selection, _image_selection(off_prefix, position=0))
    )

    with pytest.raises(ValueError, match="image selection"):
        store.issue_fork_fence(
            active.id,
            image_selections=invalid_selections,
        )


def test_stage_fork_snapshot_preserves_all_ordinary_sent_attachments() -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Source")
    user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Question",
        attachments=(
            MessageAttachment(b"alpha", "text/plain", "alpha.txt", 0),
            MessageAttachment(b"beta", "application/pdf", "beta.pdf", 1),
        ),
    )

    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(user.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )

    assert tuple(
        (item.position, item.data, item.mime_type, item.display_name)
        for item in snapshot.messages[0].attachments
    ) == (
        (0, b"alpha", "text/plain", "alpha.txt"),
        (1, b"beta", "application/pdf", "beta.pdf"),
    )


@pytest.mark.parametrize("bad_data", (None, b""))
def test_fork_rejects_missing_required_attachment_without_path_fallback(
    bad_data,
    tmp_path,
) -> None:
    source_file = tmp_path / "must-not-be-read.txt"
    source_file.write_bytes(b"filesystem fallback")
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Source")
    user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Question",
        attachments=(MessageAttachment(b"valid", "text/plain", "sent.txt", 0),),
    )
    live = store._nodes_by_session[session.id][user.id]
    live.attachments = (
        SimpleNamespace(
            data=bad_data,
            mime_type="text/plain",
            display_name="sent.txt",
            position=0,
            file_path=str(source_file),
        ),
    )

    with pytest.raises(ValueError, match="attachment"):
        store.issue_fork_fence(user.id)

    assert source_file.read_bytes() == b"filesystem fallback"


def test_stage_fork_rejects_missing_selected_generated_image_before_publication() -> (
    None
):
    store, _, session, _, _, _, selected, _ = _fork_store(generated_image=True)
    selected_live = store._nodes_by_session[session.id][selected.id]
    selection = _image_selection(selected_live, position=0)
    fence = store.issue_fork_fence(selected.id, image_selections=(selection,))
    selected_live.attachments = (replace(selected_live.attachments[0], data=None),)

    with pytest.raises(ValueError, match="source changed"):
        store.stage_fork_snapshot(
            fence,
            title="Independent fork",
            fork_session_id="fork-session",
            fork_conversation_id="fork-conversation",
        )

    assert "fork-session" not in {item.id for item in store.sessions()}


def test_stage_fork_selected_image_candidate_remains_aba_safe(monkeypatch) -> None:
    store, _, session, _, _, _, selected, _ = _fork_store(generated_image=True)
    selected_live = store._nodes_by_session[session.id][selected.id]
    selected_live.attachments = (
        *selected_live.attachments,
        MessageAttachment(
            _image_bytes((0, 255, 0)),
            "image/png",
            "selected-2.png",
            1,
        ),
    )
    selected_live.generation_metadata = (
        *selected_live.generation_metadata,
        replace(selected_live.generation_metadata[0], prompt="selected prompt"),
    )
    selection = _image_selection(selected_live, position=1)
    fence = store.issue_fork_fence(selected.id, image_selections=(selection,))
    original_attachments = selected_live.attachments
    original_validate = store.validate_fork_fence
    calls = 0

    def validate_around_transient_selected_image(candidate, *, image_selections=()):
        nonlocal calls
        calls += 1
        if calls == 1:
            valid = original_validate(candidate, image_selections=image_selections)
            selected_live.attachments = (
                original_attachments[0],
                replace(
                    original_attachments[1],
                    data=_image_bytes((0, 0, 255)),
                ),
            )
            return valid
        selected_live.attachments = original_attachments
        return original_validate(candidate, image_selections=image_selections)

    monkeypatch.setattr(
        store,
        "validate_fork_fence",
        validate_around_transient_selected_image,
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


@pytest.mark.parametrize("source_inside_snapshot", (True, False))
def test_video_projects_as_unavailable_tombstone_and_remaps_only_internal_source(
    source_inside_snapshot,
) -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Video source")
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Animate this",
    )
    source_image = store.append_generation_message(
        session.id,
        content="[image] source",
        variants=[(_image_bytes(), "image/png", _generation_meta_for_video())],
    )
    if not source_inside_snapshot:
        store.create_sibling(
            source_image.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Alternate branch",
        )
    video = store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video-key",
            prompt="animate",
            negative_prompt="none",
            backend="minimax",
            model="video-test",
            seed=8,
            duration_seconds=3.0,
            fps=24.0,
            width=640,
            height=360,
            ratio="16:9",
            source_image_message_id=source_image.id,
        ),
    )
    selections = (
        (_image_selection(source_image, position=0),) if source_inside_snapshot else ()
    )

    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(video.id, image_selections=selections),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected_video = snapshot.messages[-1]
    projected_source = next(
        (
            message
            for message in snapshot.messages
            if message.source_native_message_id == source_image.id
        ),
        None,
    )

    assert projected_video.content != video.content
    assert "unavailable" in projected_video.content.lower()
    assert "source-video-key" not in projected_video.content
    assert projected_video.attachments == ()
    assert projected_video.generation_metadata == ()
    assert projected_video.video_tombstone is not None
    assert not {
        "data",
        "bytes",
        "path",
        "file_path",
        "video_store_key",
        "cleanup_owner",
        "playable",
        "name",
    } & {field.name for field in fields(projected_video.video_tombstone)}
    expected_source = (
        projected_source.persisted_message_id if projected_source is not None else None
    )
    assert projected_video.video_tombstone.source_image_message_id == expected_source

    fork_session = store.register_fork_snapshot(snapshot, activate=False)
    fork_video = store.get_message(projected_video.native_message_id)
    assert fork_video.content == projected_video.content
    assert "source-video-key" not in fork_video.content
    assert fork_video.video_metadata is not None
    assert fork_video.video_metadata.name != "source-video-key"
    assert fork_video.video_metadata.source_image_message_id == expected_source
    assert fork_video.attachments == ()
    assert fork_session.id == "fork-session"


@pytest.mark.parametrize(
    "invalid_kind",
    (
        "marker-without-metadata",
        "metadata-without-marker",
        "marker-name-mismatch",
        "tombstone-with-live-metadata",
        "marker-with-tombstone-metadata",
    ),
)
def test_fork_issue_rejects_inconsistent_source_video_content(invalid_kind) -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Video source")
    if invalid_kind == "marker-without-metadata":
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=video_content_marker("source-video"),
        )
    elif invalid_kind == "marker-with-tombstone-metadata":
        raw = json.loads(
            VideoGenerationMetadata(
                name="source-video",
                prompt="animate",
                backend="minimax",
            ).to_json()
        )
        raw["video_generation"]["is_unavailable_tombstone"] = True
        metadata = VideoGenerationMetadata.from_json(json.dumps(raw))
        assert metadata is not None
        message = store.append_video_message(session.id, video_metadata=metadata)
    else:
        message = store.append_video_message(
            session.id,
            video_metadata=VideoGenerationMetadata(
                name="source-video",
                prompt="animate",
                backend="minimax",
            ),
        )
        live = store._nodes_by_session[session.id][message.id]
        if invalid_kind == "metadata-without-marker":
            live.content = "ordinary assistant text"
        elif invalid_kind == "tombstone-with-live-metadata":
            live.content = CONSOLE_FORK_VIDEO_TOMBSTONE_CONTENT
        else:
            live.content = video_content_marker("different-video")

    with pytest.raises(ValueError, match="video"):
        store.issue_fork_fence(message.id)


def test_registered_video_tombstone_can_be_fenced_and_forked_again() -> None:
    source_store = ConsoleChatStore()
    source = source_store.create_session(
        title="Video source",
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
        ephemeral=True,
    )
    video = source_store.append_video_message(
        source.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video",
            prompt="animate",
            backend="minimax",
        ),
    )
    first_snapshot = source_store.stage_fork_snapshot(
        source_store.issue_fork_fence(video.id),
        title="First fork",
        fork_session_id="first-fork",
        fork_conversation_id=None,
    )
    store = ConsoleChatStore()
    store.register_fork_snapshot(first_snapshot, activate=False)
    tombstone = store.get_message(first_snapshot.messages[-1].native_message_id)

    eligibility = store.fork_eligibility(tombstone.id)
    fence = store.issue_fork_fence(tombstone.id)
    second_snapshot = store.stage_fork_snapshot(
        fence,
        title="Second fork",
        fork_session_id="second-fork",
        fork_conversation_id=None,
    )

    assert eligibility.eligible is True
    assert tombstone.content == CONSOLE_FORK_VIDEO_TOMBSTONE_CONTENT
    assert tombstone.video_metadata is not None
    assert tombstone.video_metadata.is_unavailable_tombstone is True
    assert second_snapshot.messages[-1].content == CONSOLE_FORK_VIDEO_TOMBSTONE_CONTENT
    assert second_snapshot.messages[-1].video_tombstone is not None


def test_fork_rejects_a_tombstone_body_without_video_metadata() -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Invalid tombstone")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content=CONSOLE_FORK_VIDEO_TOMBSTONE_CONTENT,
    )

    with pytest.raises(ValueError, match="video"):
        store.issue_fork_fence(message.id)


def test_stage_rejects_a_forged_fence_for_mismatched_video_marker() -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Video source")
    video = store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video",
            prompt="animate",
            backend="minimax",
        ),
    )
    fence = store.issue_fork_fence(video.id)
    mismatched_content = video_content_marker("different-video")
    store._nodes_by_session[session.id][video.id].content = mismatched_content
    forged_entry = replace(fence.lineage[0], visible_content=mismatched_content)

    with pytest.raises(ValueError, match="source changed"):
        store.stage_fork_snapshot(
            replace(fence, lineage=(forged_entry,)),
            title="Independent fork",
            fork_session_id="fork-session",
            fork_conversation_id="fork-conversation",
        )


def test_video_remaps_an_ordinary_sent_source_image_inside_snapshot() -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Video source")
    source_image = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Animate this upload",
        attachments=(MessageAttachment(_image_bytes(), "image/png", "source.png", 0),),
    )
    video = store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video-key",
            prompt="animate",
            backend="minimax",
            source_image_message_id=source_image.id,
        ),
    )

    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(video.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected_source, projected_video = snapshot.messages

    assert projected_video.video_tombstone is not None
    assert (
        projected_video.video_tombstone.source_image_message_id
        == projected_source.persisted_message_id
    )


def test_video_clears_a_source_image_reference_to_a_later_message() -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Video source")
    later_image_id = "later-source-image"
    video = store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video-key",
            prompt="animate",
            backend="minimax",
            source_image_message_id=later_image_id,
        ),
    )
    later_image = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Image uploaded after the video",
        attachments=(MessageAttachment(_image_bytes(), "image/png", "source.png", 0),),
        message_id=later_image_id,
    )

    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(later_image.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected_video = snapshot.messages[0]

    assert projected_video.source_native_message_id == video.id
    assert projected_video.video_tombstone is not None
    assert projected_video.video_tombstone.source_image_message_id is None
    store.register_fork_snapshot(snapshot, activate=False)


def test_video_clears_a_source_image_reference_to_an_earlier_non_image() -> None:
    store = ConsoleChatStore()
    session = _new_fork_session(store, title="Video source")
    non_image = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Text source",
        attachments=(MessageAttachment(b"text", "text/plain", "source.txt", 0),),
    )
    video = store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video-key",
            prompt="animate",
            backend="minimax",
            source_image_message_id=non_image.id,
        ),
    )

    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(video.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected_video = snapshot.messages[-1]

    assert projected_video.video_tombstone is not None
    assert projected_video.video_tombstone.source_image_message_id is None
    store.register_fork_snapshot(snapshot, activate=False)


@pytest.mark.parametrize("tamper", ("content", "external-source"))
def test_fork_registration_rejects_tampered_video_authority(tamper) -> None:
    source_store = ConsoleChatStore()
    session = _new_fork_session(source_store, title="Video source")
    source_image = source_store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Animate this upload",
        attachments=(MessageAttachment(_image_bytes(), "image/png", "source.png", 0),),
    )
    video = source_store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video-key",
            prompt="animate",
            backend="minimax",
            source_image_message_id=source_image.id,
        ),
    )
    snapshot = source_store.stage_fork_snapshot(
        source_store.issue_fork_fence(video.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected_source, projected_video = snapshot.messages
    if tamper == "content":
        projected_video = replace(projected_video, content="[video] secret-slug")
    else:
        assert projected_video.video_tombstone is not None
        projected_video = replace(
            projected_video,
            video_tombstone=replace(
                projected_video.video_tombstone,
                source_image_message_id="external-source-id",
            ),
        )
    snapshot = replace(snapshot, messages=(projected_source, projected_video))
    registration_store = ConsoleChatStore()

    with pytest.raises(ValueError, match="video"):
        registration_store.register_fork_snapshot(snapshot, activate=False)

    assert registration_store.sessions() == []


def test_fork_registration_rejects_a_source_video_marker_without_tombstone() -> None:
    source_store = ConsoleChatStore()
    session = _new_fork_session(source_store, title="Video source")
    video = source_store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-secret",
            prompt="animate",
            backend="minimax",
        ),
    )
    snapshot = source_store.stage_fork_snapshot(
        source_store.issue_fork_fence(video.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected_video = replace(
        snapshot.messages[0],
        content=video.content,
        video_tombstone=None,
    )
    snapshot = replace(snapshot, messages=(projected_video,))
    registration_store = ConsoleChatStore()

    with pytest.raises(ValueError, match="video"):
        registration_store.register_fork_snapshot(snapshot, activate=False)

    assert registration_store.sessions() == []


def test_durable_video_tombstone_rejects_a_native_source_image_id() -> None:
    source_store = ConsoleChatStore()
    session = _new_fork_session(source_store, title="Video source")
    source_image = source_store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Animate this upload",
        attachments=(MessageAttachment(_image_bytes(), "image/png", "source.png", 0),),
    )
    video = source_store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video",
            prompt="animate",
            backend="minimax",
            source_image_message_id=source_image.id,
        ),
    )
    snapshot = source_store.stage_fork_snapshot(
        source_store.issue_fork_fence(video.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )
    projected_image, projected_video = snapshot.messages
    assert snapshot.durable is True
    assert projected_video.video_tombstone is not None
    projected_video = replace(
        projected_video,
        video_tombstone=replace(
            projected_video.video_tombstone,
            source_image_message_id=projected_image.native_message_id,
        ),
    )
    snapshot = replace(snapshot, messages=(projected_image, projected_video))
    registration_store = ConsoleChatStore()

    with pytest.raises(ValueError, match="video"):
        registration_store.register_fork_snapshot(snapshot, activate=False)

    assert registration_store.sessions() == []


def test_fork_registration_preserves_ordinary_prose_that_mentions_video() -> None:
    source_store = ConsoleChatStore()
    session = _new_fork_session(source_store, title="Ordinary source")
    message = source_store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="This video essay needs a concise summary.",
    )
    snapshot = source_store.stage_fork_snapshot(
        source_store.issue_fork_fence(message.id),
        title="Independent fork",
        fork_session_id="fork-session",
        fork_conversation_id="fork-conversation",
    )

    registration_store = ConsoleChatStore()
    registration_store.register_fork_snapshot(snapshot, activate=False)

    assert (
        registration_store.get_message(snapshot.messages[0].native_message_id).content
        == message.content
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
        thinking_history_policy=session.thinking_history_policy,
    )


def test_temporary_fork_preserves_thinking_history_policy_and_invalidates_fence() -> (
    None
):
    store, _, session, _, _, _, selected, _ = _fork_store(ephemeral=True)
    store.set_session_thinking_history_policy(session.id, "include")
    fence = store.issue_fork_fence(selected.id)
    snapshot = store.stage_fork_snapshot(
        fence,
        title="Policy fork",
        fork_session_id="policy-fork",
        fork_conversation_id=None,
    )

    assert snapshot.configuration.thinking_history_policy == "include"
    registered = ConsoleChatStore().register_fork_snapshot(snapshot, activate=False)
    assert registered.thinking_history_policy == "include"

    store.set_session_thinking_history_policy(session.id, "exclude")
    assert store.validate_fork_fence(fence) is False


def test_persist_message_if_needed_hides_durable_publish_gap_from_fork() -> None:
    started = Event()
    release = Event()

    class BlockingPersistence(_ForkVersionPersistence):
        def create_message(self, **_kwargs):
            started.set()
            assert release.wait(5)
            return "persisted-deferred"

    persistence = BlockingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session()
    session.persisted_conversation_id = "conv-1"
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Deferred",
        persist=False,
    )
    result = []
    worker = Thread(
        target=lambda: result.append(store.persist_message_if_needed(message.id))
    )
    worker.start()
    assert started.wait(5)

    try:
        eligibility = store.fork_eligibility(message.id)
        assert eligibility.eligible is False
        assert "changing" in eligibility.reason.lower()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert result[0].persisted_message_id is not None


def test_mark_message_send_blocked_marks_transition_before_live_status_change(
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Blocked",
        persist=False,
    )
    started = Event()
    release = Event()
    original_bump = store._bump_message_speech_revision

    def blocking_bump(message_id: str) -> int:
        started.set()
        assert release.wait(5)
        return original_bump(message_id)

    monkeypatch.setattr(store, "_bump_message_speech_revision", blocking_bump)
    worker = Thread(target=store.mark_message_send_blocked, args=(message.id,))
    worker.start()
    assert started.wait(5)

    try:
        eligibility = store.fork_eligibility(message.id)
        assert eligibility.eligible is False
        assert "changing" in eligibility.reason.lower()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert store.get_message(message.id).status == "failed"


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
        durable_policy = ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        )
        base_snapshot = _registration_snapshot()
        snapshot = replace(
            base_snapshot,
            configuration=replace(
                base_snapshot.configuration,
                library_policy=durable_policy,
            ),
        )
        conversation_id = service.create_conversation(
            conversation_id=snapshot.fork_conversation_id,
            conversation_title=snapshot.title,
        )
        assert conversation_id == snapshot.fork_conversation_id
        seeded = service.console_library_policy_repository.insert(
            conversation_id,
            durable_policy,
        )
        assert seeded.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
        assert seeded.snapshot.policy_revision == 1
        for message in snapshot.messages:
            assert message.persisted_message_id is not None
            service.create_message(
                conversation_id=conversation_id,
                sender=message.role.value,
                content=message.content,
                message_id=message.persisted_message_id,
                parent_message_id=message.persisted_parent_id,
            )
        db.set_conversation_active_leaf(
            conversation_id,
            snapshot.messages[-1].persisted_message_id,
        )

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
        boundary_id = snapshot.messages[-1].native_message_id
        assert store.fork_eligibility(boundary_id) == ConsoleForkEligibility(
            False,
            "Durable Console Library policy is not loaded.",
        )
        with pytest.raises(ValueError, match="Library policy is not loaded"):
            store.issue_fork_fence(boundary_id)

        hydrated = await store.hydrate_session_library_policy(session.id)
        assert session.library_policy_hydrated is True
        assert hydrated.source == "durable"
        assert hydrated.policy_revision == 1
        assert hydrated.auto_retrieve is durable_policy.auto_retrieve
        assert hydrated.assistant_access is durable_policy.assistant_access
        assert store.fork_eligibility(boundary_id).eligible is True
        fence = store.issue_fork_fence(boundary_id)
        refork = store.stage_fork_snapshot(
            fence,
            title="Independent refork",
            fork_session_id="refork-session",
            fork_conversation_id="refork-conversation",
        )
        assert refork.configuration.library_policy == durable_policy

        edited = ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
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
