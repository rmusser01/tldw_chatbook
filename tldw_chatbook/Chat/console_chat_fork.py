"""Pure immutable contracts for projecting one Console chat into a fork."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleMessageStatus,
    derive_console_session_title,
)
from tldw_chatbook.Chat.console_context_policy import (
    CompactionFailureBehavior,
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCarryForwardMode,
    ContextCompactionMode,
    ContextCompactionRepresentation,
)
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


CONSOLE_FORK_TITLE_MAX_LENGTH = 60
CONSOLE_FORK_FINGERPRINT_JSON_MAX_BYTES = 64 * 1024
_CONSOLE_FORK_FINGERPRINT_DOMAIN = b"tldw_chatbook.console.chat-fork.v1\0"

ConsoleForkDurability = Literal["temporary", "durable", "unsaved_persistable"]
ConsoleForkCitationState = Literal["active_required", "unavailable", "none"]


@dataclass(frozen=True, slots=True)
class ConsoleForkEligibility:
    """Store-approved availability for one captured fork boundary."""

    eligible: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ConsoleForkLineageFence:
    """Stable source facts for one message in a captured active lineage."""

    native_message_id: str
    persisted_message_id: str | None
    native_parent_id: str | None
    role: ConsoleMessageRole
    status: ConsoleMessageStatus
    visible_content: str
    visible_variant_id: str | None
    sibling_identity: tuple[str, ...]
    persisted_revision: int | None
    attachment_fingerprint: str


@dataclass(frozen=True, slots=True)
class ConsoleForkImageSelectionFence:
    """Screen-owned selected generated-image position and payload token."""

    native_message_id: str
    selected_position: int
    browse_revision: int
    attachment_meta_fingerprint: str


@dataclass(frozen=True, slots=True)
class ConsoleForkFence:
    """Immutable source identity and active-lineage fence opened by the dialog."""

    source_session_id: str
    source_conversation_id: str | None
    source_conversation_version: int | str | None
    source_durability: ConsoleForkDurability
    source_title: str
    source_configuration_fingerprint: str
    boundary_message_id: str
    lineage: tuple[ConsoleForkLineageFence, ...]
    image_selections: tuple[ConsoleForkImageSelectionFence, ...]


@dataclass(frozen=True, slots=True)
class ConsoleForkProjectedMessage:
    """One independently owned USER/ASSISTANT message in a fork snapshot."""

    native_message_id: str
    persisted_message_id: str | None
    native_parent_id: str | None
    persisted_parent_id: str | None
    turn_id: str | None
    role: ConsoleMessageRole
    status: ConsoleMessageStatus
    content: str


@dataclass(frozen=True, slots=True)
class ConsoleForkConfigurationSnapshot:
    """Allowlisted declarative configuration for future fork turns."""

    workspace_id: str
    settings: ConsoleSessionSettings
    rag_scope: RagScope | None
    context_policy_overrides: ConsoleContextPolicyOverrides
    library_policy: ConsoleLibraryPolicyCandidate
    runtime_backend: str
    assistant_kind: str | None
    assistant_id: str | None
    assistant_authority_id: str | None
    persona_memory_mode: str | None
    character_id: int | None
    character_name: str | None
    user_display_name_override: str | None
    character_system_template: str | None
    speech_preferences: ConsoleSpeechPreferences
    project_instruction_state: ProjectInstructionControlState


@dataclass(frozen=True, slots=True)
class ConsoleForkCitationLink:
    """Required governed-citation state for one copied durable message."""

    source_persisted_message_id: str
    source_revision: int
    state: ConsoleForkCitationState


@dataclass(frozen=True, slots=True)
class ConsoleChatForkSnapshot:
    """Complete immutable input for durable commit or detached publication."""

    fork_session_id: str
    fork_conversation_id: str | None
    title: str
    source_session_id: str
    source_conversation_id: str | None
    source_boundary_persisted_message_id: str | None
    durable: bool
    messages: tuple[ConsoleForkProjectedMessage, ...]
    configuration: ConsoleForkConfigurationSnapshot
    citation_links: tuple[ConsoleForkCitationLink, ...]


def normalize_fork_title(title: str) -> str:
    """Return one nonblank Console title bounded by the shared title helper.

    Args:
        title: Proposed user-visible fork title.

    Returns:
        The collapsed title capped at ``CONSOLE_FORK_TITLE_MAX_LENGTH``.

    Raises:
        ValueError: If the normalized title is blank.
    """

    normalized = derive_console_session_title(
        title,
        max_length=CONSOLE_FORK_TITLE_MAX_LENGTH,
    )
    if not normalized:
        raise ValueError("Fork title cannot be blank.")
    return normalized


def default_fork_title(source_title: str) -> str:
    """Return the bounded default title for a fork naming dialog.

    Args:
        source_title: Current source-chat display title.

    Returns:
        A bounded ``Forked from …`` title or the approved untitled fallback.
    """

    source = " ".join(str(source_title or "").split())
    candidate = f"Forked from {source}" if source else "Untitled chat — fork"
    return normalize_fork_title(candidate)


def fingerprint_console_fork_configuration(
    configuration: ConsoleForkConfigurationSnapshot,
) -> str:
    """Fingerprint only the declarative configuration allowed into a fork.

    Args:
        configuration: Typed configuration captured for future fork turns.

    Returns:
        A lowercase SHA-256 hex digest.

    Raises:
        TypeError: If the value is not the narrow configuration contract.
        ValueError: If source-only notice authority or pinned prefill remains.
    """

    if type(configuration) is not ConsoleForkConfigurationSnapshot:
        raise TypeError("Expected ConsoleForkConfigurationSnapshot.")
    _require_exact_field_types(
        "Fork configuration",
        configuration,
        {
            "workspace_id": (str,),
            "runtime_backend": (str,),
            "assistant_kind": (str, type(None)),
            "assistant_id": (str, type(None)),
            "assistant_authority_id": (str, type(None)),
            "persona_memory_mode": (str, type(None)),
            "character_id": (int, type(None)),
            "character_name": (str, type(None)),
            "user_display_name_override": (str, type(None)),
            "character_system_template": (str, type(None)),
        },
    )
    if type(configuration.settings) is not ConsoleSessionSettings:
        raise TypeError("Fork settings must be ConsoleSessionSettings.")
    _require_exact_field_types(
        "Fork configuration settings",
        configuration.settings,
        {
            "provider": (str,),
            "model": (str, type(None)),
            "base_url": (str, type(None)),
            "temperature": (int, float),
            "top_p": (int, float),
            "min_p": (int, float, type(None)),
            "top_k": (int, type(None)),
            "max_tokens": (int, type(None)),
            "seed": (int, type(None)),
            "presence_penalty": (int, float, type(None)),
            "frequency_penalty": (int, float, type(None)),
            "reasoning_effort": (str, type(None)),
            "reasoning_summary": (str, type(None)),
            "verbosity": (str, type(None)),
            "thinking_effort": (str, type(None)),
            "thinking_budget_tokens": (int, type(None)),
            "streaming": (bool,),
            "character_label": (str,),
            "system_prompt": (str, type(None)),
            "source": (str,),
            "pinned_prefill": (str, type(None)),
        },
    )
    if configuration.settings.pinned_prefill is not None:
        raise ValueError("Fork configuration must clear pinned prefill.")
    if type(configuration.rag_scope) not in {RagScope, type(None)}:
        raise TypeError("Fork RAG scope must be RagScope or None.")
    if configuration.rag_scope is not None:
        _require_exact_field_types(
            "Fork configuration RAG scope",
            configuration.rag_scope,
            {
                "items": (tuple,),
                "updated_at": (str,),
                "empty_is_scoped": (bool,),
            },
        )
        for item in configuration.rag_scope.items:
            if type(item) is not ScopeItem:
                raise TypeError("Fork configuration RAG items must be ScopeItem.")
            _require_exact_field_types(
                "Fork configuration RAG item",
                item,
                {"source_type": (str,), "source_id": (str,)},
            )
    if type(configuration.context_policy_overrides) is not ConsoleContextPolicyOverrides:
        raise TypeError(
            "Fork context policy must be ConsoleContextPolicyOverrides."
        )
    _require_exact_field_types(
        "Fork configuration context policy",
        configuration.context_policy_overrides,
        {
            "budget_mode": (ContextBudgetMode, type(None)),
            "custom_budget_tokens": (int, type(None)),
            "compaction_mode": (ContextCompactionMode, type(None)),
            "compaction_representation": (
                ContextCompactionRepresentation,
                type(None),
            ),
            "trigger_ratio": (int, float, type(None)),
            "target_ratio": (int, float, type(None)),
            "summary_max_tokens": (int, type(None)),
            "failure_behavior": (CompactionFailureBehavior, type(None)),
            "carry_forward_mode": (ContextCarryForwardMode, type(None)),
        },
    )
    if type(configuration.library_policy) is not ConsoleLibraryPolicyCandidate:
        raise TypeError("Fork library policy must be ConsoleLibraryPolicyCandidate.")
    _require_exact_field_types(
        "Fork configuration library policy",
        configuration.library_policy,
        {
            "auto_retrieve": (ConsoleAutoRetrieve,),
            "assistant_access": (ConsoleAssistantLibraryAccess,),
        },
    )
    if type(configuration.speech_preferences) is not ConsoleSpeechPreferences:
        raise TypeError("Fork speech preferences must be ConsoleSpeechPreferences.")
    _require_exact_field_types(
        "Fork configuration speech preferences",
        configuration.speech_preferences,
        {
            "auto_speak": (bool,),
            "paused": (bool,),
            "consent_destination": (str, type(None)),
            "consent_version": (int,),
        },
    )
    project_state = configuration.project_instruction_state
    if type(project_state) is not ProjectInstructionControlState:
        raise TypeError(
            "Fork project instructions must be ProjectInstructionControlState."
        )
    if project_state.project_instruction_notice_key is not None:
        raise ValueError("Fork configuration must clear source notice authority.")
    project_state = sanitize_fork_project_instruction_state(project_state)

    settings = configuration.settings
    rag_scope = configuration.rag_scope
    payload = {
        "workspace_id": configuration.workspace_id,
        "settings": {
            "provider": settings.provider,
            "model": settings.model,
            "base_url": settings.base_url,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "min_p": settings.min_p,
            "top_k": settings.top_k,
            "max_tokens": settings.max_tokens,
            "seed": settings.seed,
            "presence_penalty": settings.presence_penalty,
            "frequency_penalty": settings.frequency_penalty,
            "reasoning_effort": settings.reasoning_effort,
            "reasoning_summary": settings.reasoning_summary,
            "verbosity": settings.verbosity,
            "thinking_effort": settings.thinking_effort,
            "thinking_budget_tokens": settings.thinking_budget_tokens,
            "streaming": settings.streaming,
            "character_label": settings.character_label,
            "system_prompt": settings.system_prompt,
            "source": settings.source,
            "pinned_prefill": None,
        },
        "rag_scope": (
            None
            if rag_scope is None
            else {
                "items": [
                    {
                        "source_type": item.source_type,
                        "source_id": item.source_id,
                    }
                    for item in rag_scope.items
                ],
                "updated_at": rag_scope.updated_at,
                "empty_is_scoped": rag_scope.empty_is_scoped,
            }
        ),
        "context_policy_overrides": configuration.context_policy_overrides.to_dict(),
        "library_policy": {
            "auto_retrieve": configuration.library_policy.auto_retrieve.value,
            "assistant_access": configuration.library_policy.assistant_access.value,
        },
        "runtime_backend": configuration.runtime_backend,
        "assistant_kind": configuration.assistant_kind,
        "assistant_id": configuration.assistant_id,
        "assistant_authority_id": configuration.assistant_authority_id,
        "persona_memory_mode": configuration.persona_memory_mode,
        "character_id": configuration.character_id,
        "character_name": configuration.character_name,
        "user_display_name_override": configuration.user_display_name_override,
        "character_system_template": configuration.character_system_template,
        "speech_preferences": {
            "auto_speak": configuration.speech_preferences.auto_speak,
            "paused": configuration.speech_preferences.paused,
            "consent_destination": configuration.speech_preferences.consent_destination,
            "consent_version": configuration.speech_preferences.consent_version,
        },
        "project_instruction_state": {
            "project_instructions_enabled": project_state.project_instructions_enabled,
            "working_folder_binding_id": project_state.working_folder_binding_id,
            "working_folder_locator_fingerprint": (
                project_state.working_folder_locator_fingerprint
            ),
            "project_instruction_notice_key": None,
        },
    }
    return _fingerprint_console_fork_payload("configuration", payload)


def fingerprint_console_fork_image_selection(
    image_selection: ConsoleForkImageSelectionFence,
) -> str:
    """Fingerprint one allowlisted generated-image selection fence."""

    if type(image_selection) is not ConsoleForkImageSelectionFence:
        raise TypeError("Expected ConsoleForkImageSelectionFence.")
    _require_exact_field_types(
        "Fork image selection",
        image_selection,
        {
            "native_message_id": (str,),
            "selected_position": (int,),
            "browse_revision": (int,),
            "attachment_meta_fingerprint": (str,),
        },
    )
    return _fingerprint_console_fork_payload(
        "image-selection",
        {
            "native_message_id": image_selection.native_message_id,
            "selected_position": image_selection.selected_position,
            "browse_revision": image_selection.browse_revision,
            "attachment_meta_fingerprint": (
                image_selection.attachment_meta_fingerprint
            ),
        },
    )


def _fingerprint_console_fork_payload(purpose: str, payload: object) -> str:
    """Hash one internal allowlisted payload as bounded canonical JSON."""

    _validate_canonical_json(payload)
    try:
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except ValueError as exc:
        raise TypeError("Fork fingerprint payload must be canonical JSON.") from exc
    if len(canonical) > CONSOLE_FORK_FINGERPRINT_JSON_MAX_BYTES:
        raise ValueError("Fork fingerprint payload must remain bounded.")
    return hashlib.sha256(
        _CONSOLE_FORK_FINGERPRINT_DOMAIN + purpose.encode("ascii") + b"\0" + canonical
    ).hexdigest()


def _require_exact_field_types(
    contract: str,
    value: object,
    field_types: dict[str, tuple[type, ...]],
) -> None:
    """Reject containers and bool-as-int values in typed fingerprint leaves."""

    for field_name, allowed_types in field_types.items():
        if type(getattr(value, field_name)) not in allowed_types:
            raise TypeError(f"{contract} {field_name} has an invalid type.")


def _validate_canonical_json(value: object) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_canonical_json(item)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for item in value.values():
            _validate_canonical_json(item)
        return
    raise TypeError("Fork fingerprint payload must be canonical JSON.")
