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
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicyCandidate
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences
from tldw_chatbook.Chat.rag_scope import RagScope


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


def fingerprint_console_fork_payload(purpose: str, payload: object) -> str:
    """Hash one bounded canonical-JSON payload under a fork-specific purpose.

    Args:
        purpose: Short domain label such as ``configuration``.
        payload: JSON-only data selected by the fork allowlist.

    Returns:
        A lowercase SHA-256 hex digest.

    Raises:
        TypeError: If ``payload`` is not canonical JSON data.
        ValueError: If the purpose or serialized payload exceeds its bound.
    """

    if (
        not isinstance(purpose, str)
        or not purpose
        or len(purpose) > 64
        or not purpose.isascii()
        or any(character.isspace() for character in purpose)
    ):
        raise ValueError("Fork fingerprint purpose must be bounded ASCII text.")
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
