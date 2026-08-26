"""Native Console chat session store and persistence facade."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Protocol, Sequence
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Character_Chat.character_mood import detect_character_mood
from tldw_chatbook.Character_Chat.emote_directives import (
    CharacterEmoteEvent,
    CharacterEmoteRunSnapshot,
    CharacterEmoteStreamParser,
    utf16_length,
)
from tldw_chatbook.Agents.agent_models import (
    FinalContinuation,
    ProviderContinuationEvent,
    ToolBatchReady,
    ToolCallExecuting,
    ToolCallFinished,
)
from tldw_chatbook.Agents.session_todo_store import SessionTodoStore
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.DB.ChaChaNotes_DB import TrajectoryRowWrite
from tldw_chatbook.Chat.citation_trace_models import (
    ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX,
    SealedCitationWrite,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
)
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    CONSOLE_EPHEMERAL_PROMOTION_BLOCK_COPY,
    DEFAULT_CONSOLE_SESSION_TITLE,
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleCitationPresentation,
    ConsoleMessageFeedback,
    ConsoleMessageRole,
    ConsoleMessageStatus,
    ConsoleDispatchRecoveryActionId,
    ConsoleDispatchRecoveryKind,
    ConsoleDispatchRecoveryState,
    ConsoleVariant,
    ConsoleVariantSet,
    ConsoleWorkspaceContext,
    GenerationVariantMeta,
    MessageAttachment,
    console_dispatch_recovery_from_checkpoint,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleAssistantSettlement,
    ConsoleContinuationHandoff,
    ConsoleDispatchCheckpoint,
    ConsoleDispatchCheckpointState,
    ConsoleDispatchReconstructability,
    ConsoleDispatchResultStatus,
    ConsoleDispatchTransition,
    ConsoleDurableTurnAcceptance,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_destination import (
    ConsoleLibraryDestinationRuntimeState,
    settle_console_library_destination_runtime,
    update_console_library_destination_runtime,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyDefaults,
    ConsoleLibraryPolicyHolder,
    ConsoleLibraryPolicySnapshot,
    ConsoleLibraryPolicyWriteResult,
    normalize_policy_read,
)
from tldw_chatbook.Chat.console_library_policy_coordinator import (
    ConsoleLibraryPolicyCoordinator,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
    apply_preparation_transition,
)
from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleTransactionContribution,
)
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail, capture_to_blob
from tldw_chatbook.Chat.console_capture_policy_repository import (
    CapturePolicyReadResult,
    CapturePolicyReadStatus,
    CapturePolicyWriteStatus,
    ConsoleCapturePolicyRepository,
)
from tldw_chatbook.Chat.console_roleplay_identity import (
    ConsolePresentationContext,
    effective_user_display_name,
    expand_character_template,
    normalize_chat_display_name,
    resolve_console_message_presentation,
)
from tldw_chatbook.Chat.console_roleplay_metadata import (
    ConsoleRoleplayContext,
    merge_console_roleplay_context,
)
from tldw_chatbook.Chat.console_prefill import PINNED_PREFILL_METADATA_KEY
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    decode_project_context_json,
    encode_project_context_json,
)
from tldw_chatbook.Chat.console_speech import (
    ConsoleSpeechSnapshotRejected,
    ConsoleSpeechSnapshotRejectionCode,
    TTSMessageSpeechSnapshot,
)
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences
from tldw_chatbook.Chat.message_metadata import (
    CharacterEmoteEventMetadata,
    CharacterEmoteMetadata,
    MessageMetadata,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.thinking_blocks import (
    ThinkingEnvelope,
    ThinkingStatus,
    dump_thinking_blocks_json,
    read_thinking_blocks_json,
)
from tldw_chatbook.Chat.trajectory import contains_local_path
from tldw_chatbook.Chat.provider_continuation import (
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
    read_provider_continuation_json,
    transition_provider_call,
)
from tldw_chatbook.Chat.rag_scope import RagScope, SessionScopeHolder, serialize_scope
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.TTS.profile_errors import ProfileValidationError
from tldw_chatbook.TTS.profile_types import CharacterRef
from tldw_chatbook.Utils.log_sanitizer import REDACTION_MARKER, redact_log_line
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Video_Generation.video_store import video_content_marker

if TYPE_CHECKING:
    # Annotation-only: ``from __future__ import annotations`` (top of file)
    # defers evaluation of every type hint below, so this class never needs
    # to exist at runtime here -- only ``capture_to_blob`` (imported above)
    # is actually called.
    from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture

#: Maximum number of attachments a Console session may stage before send.
MAX_PENDING_ATTACHMENTS = 5

TerminalCitationFinalizer = Callable[[str], SealedCitationWrite | None]


@dataclass(frozen=True, slots=True)
class CharacterEmoteLiveEvent:
    """Content-free process-local expression event for a Console session."""

    sequence: int
    session_id: str
    message_id: str
    state: str


@dataclass(slots=True)
class _CharacterEmoteCapture:
    """Ephemeral parser state for one armed assistant generation."""

    parser: CharacterEmoteStreamParser
    snapshot: CharacterEmoteRunSnapshot
    events: list[CharacterEmoteEvent] = field(default_factory=list)
    fail_closed: bool = False


class ConsoleDispatchSettlementError(RuntimeError):
    """An owned dispatch terminal could not commit atomically."""


class ConsoleDurableAcceptanceRetired(RuntimeError):
    """The preparation was retired underneath an in-flight postcommit effect.

    TASK-22587: closing a Console session retires its durable preparation, so
    an effect still in flight finds its fingerprint gone. That is an ORDINARY
    consequence of the user closing a chat, and it is not the same event as a
    fingerprint that changed unexpectedly -- which is a bug, and which must
    keep raising the bare ``RuntimeError`` the postcommit APIs already document.

    Retirement is decidable rather than inferred: ``retire_durable_acceptance``
    leaves a tombstone carrying the SAME fingerprint, so a matching tombstone
    proves the preparation was retired rather than mutated.
    """


class ConsoleThinkingCompatibilityError(RuntimeError):
    """A generation mutation would replace unreadable durable thinking."""


def require_thinking_persistence_support(
    persistence: ConsoleChatPersistence | None,
    *,
    persistent: bool,
    may_emit_thinking: bool,
) -> None:
    """Fail before send when a durable backend cannot round-trip thinking V1."""
    if not persistent or not may_emit_thinking:
        return
    version_reader = getattr(persistence, "thinking_round_trip_version", None)
    version = version_reader() if callable(version_reader) else None
    if type(version) is not int or version != 1:
        raise ConsoleThinkingCompatibilityError(
            "This persistent backend cannot preserve model thinking version 1. "
            "Upgrade it before sending."
        )


def _refuse_roleplay_projection_write(**_kwargs: object) -> bool:
    """Represent a missing durable projection seam in an immutable plan."""
    return False


@dataclass(frozen=True)
class _VariantStreamBase:
    """Pre-regenerate snapshot captured by ``begin_variant_stream``.

    Carries the visible content, the message's status *and* its recorded
    usage at the moment regeneration began, so a failed or stopped
    regenerate can restore the message to exactly the state it was in
    before -- not just its content. Usage is part of that state: it
    describes the generation that produced the content being restored, so
    leaving the abandoned run's numbers behind would attribute one
    generation's spend to another's answer.
    """

    content: str
    prior_status: ConsoleMessageStatus
    prior_usage: "ProviderUsage | None" = None
    prior_metadata: MessageMetadata | None = None
    prior_thinking: ThinkingEnvelope | None = field(default=None, repr=False)
    prior_opaque_thinking_json: str | None = field(default=None, repr=False)
    prior_thinking_warning: str | None = None
    prior_thinking_actions_enabled: bool = True
    prior_provider_continuation: ProviderContinuationCheckpoint | None = field(
        default=None, repr=False
    )
    prior_provider_continuation_warning: str | None = None
    prior_provider_continuation_remote: bool = False
    prior_provider_continuation_actions_enabled: bool = True
    prior_assistant_generation_state: str | None = None


@dataclass(frozen=True, slots=True)
class _RoleplaySystemPromptWrite:
    writer: Callable[..., object] = field(repr=False, compare=False)
    conversation_id: str
    system_prompt: str | None
    expected_roleplay_context: ConsoleRoleplayContext
    expected_system_prompts: tuple[str | None, ...]
    accepts_roleplay_context_guard: bool
    accepts_system_prompt_guard: bool
    accepts_source_owned_repair: bool
    source_owned_repair: bool
    accepts_roleplay_version_guard: bool
    expected_roleplay_version: int | None


@dataclass(frozen=True, slots=True)
class _RoleplaySyncWrite:
    writer: Callable[..., object] = field(repr=False, compare=False)
    stable_key: str
    kwargs: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _RoleplayMessageProjectionWrite:
    writer: Callable[..., object] = field(repr=False, compare=False)
    native_message_id: str
    message_id: str
    content: str
    image_data: bytes | None
    image_mime_type: str | None
    feedback: ConsoleMessageFeedback | None
    metadata_json: str | None
    accepts_attachments: bool
    accepts_metadata_json: bool
    expected_roleplay_template_source: str
    expected_message_contents: tuple[str, ...]
    accepts_template_source_guard: bool
    accepts_message_contents_guard: bool
    accepts_source_owned_repair: bool
    source_owned_repair: bool
    accepts_roleplay_version_guard: bool
    expected_roleplay_version: int | None
    sync_write: _RoleplaySyncWrite | None = None


@dataclass(frozen=True, slots=True)
class _RoleplayMessageProjectionPersistenceOutcome:
    """One durable greeting outcome plus its owner-accepted Sync intent."""

    native_message_id: str
    content: str
    persisted: bool
    sync_write: _RoleplaySyncWrite | None = None


@dataclass(frozen=True, slots=True)
class ConsoleRoleplayProjectionPersistencePlan:
    """Immutable durable writes prepared after owner-thread materialization."""

    session_id: str
    generation: int
    system_prompt_write: _RoleplaySystemPromptWrite | None
    message_writes: tuple[_RoleplayMessageProjectionWrite, ...]


@dataclass(frozen=True, slots=True)
class ConsoleRoleplayProjectionPersistenceResult:
    """Immutable outcome returned by an off-thread projection-plan consumer."""

    session_id: str
    generation: int
    persisted: bool
    system_prompt_attempted: bool
    system_prompt: str | None
    system_prompt_persisted: bool
    message_outcomes: tuple[_RoleplayMessageProjectionPersistenceOutcome, ...] = ()


@dataclass(frozen=True, slots=True)
class ConsoleStagedConversationIdentity:
    """Preallocated durable identity published only after transaction commit."""

    conversation_id: str
    title: str


@dataclass(frozen=True, slots=True)
class ConsoleDurableTurnCommit:
    """Immutable durable values returned before live publication."""

    identity: ConsoleStagedConversationIdentity
    user_message_id: str
    user_message_version: int
    assistant_message_id: str
    assistant_message_version: int
    checkpoint: ConsoleDispatchCheckpoint


@dataclass(frozen=True, slots=True)
class ConsoleDurablePostcommitEffects:
    """App-lifetime completion ledger for one committed preparation."""

    preparation_id: str
    session_id: str
    assistant_message_id: str
    completed: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ConsoleDurableAcceptanceFingerprint:
    """Body-free immutable owner for one app-lifetime durable acceptance."""

    preparation_id: str
    session_id: str
    conversation_id: str
    title_hash: str
    attempt_id: str
    origin: str
    queue_entry_id: str | None
    user_message_id: str
    assistant_message_id: str
    digest: str


@dataclass(frozen=True, slots=True)
class _ConsoleStagedDurableOwnerIds:
    """Stable USER/assistant identities reserved for every commit retry."""

    user_message_id: str
    assistant_message_id: str


@dataclass(frozen=True, slots=True)
class _ConsoleDurableCommitReservation:
    """Unique caller ownership installed before acceptance canonicalization."""

    caller_token: object
    owner_thread_id: int
    preparation_id: str
    attempt_id: str
    session_id: str
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    origin: str
    queue_entry_id: str | None


@dataclass(frozen=True, slots=True)
class _ConsoleDurableTombstone:
    """Bounded content-free proof that one acceptance owner was retired."""

    fingerprint: ConsoleDurableAcceptanceFingerprint
    completed: frozenset[str]


def _default_library_policy_holder() -> ConsoleLibraryPolicyHolder:
    return ConsoleLibraryPolicyHolder(
        ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=None,
            source="new_session",
        )
    )


class ConsoleChatPersistence(Protocol):
    """Persistence surface used by Console without importing DB dependencies."""

    #: Raw DB handle backing this persistence adapter, or ``None`` when the
    #: adapter has none (e.g. a test fake, or a future persistence shape
    #: with no single underlying database). ``persist_session_if_needed``
    #: reaches through this seam -- rather than an undeclared ``getattr``
    #: probe -- to flush a session-held RAG retrieval scope
    #: (``SessionScopeHolder``) at first persistence (PR #747 review: a
    #: conforming adapter that structurally satisfied this Protocol without
    #: declaring ``.db`` made the flush silently no-op, losing the user's
    #: pre-persistence scope selection with no diagnostic). Declaring it
    #: here makes the seam an explicit, checkable part of the contract.
    db: Any | None

    def thinking_round_trip_version(self) -> int:
        """Return the exact supported durable thinking envelope version."""

    def commit_durable_turn(
        self,
        *,
        acceptance: ConsoleDurableTurnAcceptance,
        policy_candidate: ConsoleLibraryPolicyCandidate,
        conversation_kwargs: Mapping[str, object],
    ) -> ConsoleDispatchCheckpoint:
        """Atomically create/validate and accept one durable Console turn."""

    def create_conversation(self, **kwargs) -> str:
        """Create a persisted conversation and return its ID."""

    def create_message(
        self,
        *,
        conversation_id: str,
        sender: str,
        content: str,
        image_data: bytes | None,
        image_mime_type: str | None,
        message_id: str | None = None,
        parent_message_id: str | None = None,
        feedback: str | None = None,
        attachments: Sequence[Mapping[str, Any]] | None = None,
        citation_write: SealedCitationWrite | None = None,
        usage_json: str | None = None,
        metadata_json: str | None = None,
        thinking_blocks_json: str | None = None,
        provider_continuation_json: str | None = None,
        assistant_generation_state: str | None = None,
    ) -> str:
        """Create a persisted message and return its ID.

        ``attachments``, when given, covers ALL positions (0..N-1) and is
        authoritative over the scalar ``image_data``/``image_mime_type``
        kwargs; ``None`` leaves the pre-split legacy behavior unchanged.
        Optional: fakes used in tests may omit this parameter entirely.

        ``citation_write``, when present, is committed atomically with the
        message by citation-aware adapters. Narrow test fakes may omit this
        optional parameter entirely.

        ``usage_json`` (Console cost ticker), when present, is the
        message's normalized provider-usage JSON. Optional: narrow test
        fakes may omit this parameter entirely -- the store only passes it
        to adapters that declare it (see ``_persistence_accepts_kwarg``).

        ``metadata_json`` (task-2364), when present, is the message's
        structured metadata JSON (engine provenance, interrupted flag,
        transcript status). Same optionality and same declare-to-receive
        rule as ``usage_json``.

        The three assistant-generation fields are optional for narrow fakes,
        but production adapters receive them in the same create transaction.
        """

    def update_message_content(
        self,
        *,
        message_id: str,
        content: str,
        image_data: bytes | None,
        image_mime_type: str | None,
        parent_message_id: str | None = None,
        feedback: str | None = None,
        update_parent: bool = False,
        update_feedback: bool = False,
        attachments: Sequence[Mapping[str, Any]] | None = None,
        usage_json: str | None = None,
        metadata_json: str | None = None,
        preserve_provider_continuation: bool = False,
        clear_generation_provenance: bool = False,
    ) -> bool:
        """Update persisted message content.

        ``attachments`` follows the same split-addressing contract as
        ``create_message``; ``None`` (the Console store's edit path always
        passes this) leaves attachments untouched. Optional: fakes used in
        tests may omit this parameter entirely.

        ``usage_json`` (Console cost ticker), when present, overwrites the
        row's normalized provider-usage JSON. Optional: narrow test fakes
        may omit this parameter entirely -- the store only passes it to
        adapters that declare it, and only when usage is actually known,
        so a content-only update never clobbers an existing value with
        ``None``.

        ``metadata_json`` (task-2364) follows the identical contract for
        the structured metadata column.
        """

    def replace_assistant_generation_projection(
        self,
        *,
        message_id: str,
        content: str,
        thinking_blocks_json: str | None,
        provider_continuation_json: str | None,
        assistant_generation_state: str | None,
        usage_json: str | None,
        expected_version: int | None = None,
    ) -> int:
        """Atomically replace one selected assistant generation."""

    def update_message_usage(self, *, message_id: str, usage_json: str) -> bool:
        """Persist normalized usage as a version-neutral, local-only write.

        Unlike ``update_message_content``'s optional ``usage_json`` kwarg
        (which rides a content update and legitimately bumps the row's
        version), this method exists SOLELY for a usage-only flush against
        an already-terminal message -- the Stop-path case described on
        ``ConsoleChatStore.set_message_usage``. It must not advance
        ``version``/``last_modified`` (the ``messages_sync_update`` trigger
        watches those columns, not just content, so bumping them on a
        usage-only write would enqueue a ``sync_log`` row whose payload can
        never carry ``usage_json`` -- pure cross-device churn for a column
        that is local-only by design).

        Entirely optional: this whole method, not just a kwarg, may be
        absent. The store probes for it with ``hasattr``/``callable``
        (same philosophy as ``_persistence_accepts_kwarg``) and falls back
        to the ordinary content-carrying update path when it is not
        present, so narrow test fakes written before this method existed
        keep working unchanged.
        """

    def update_message_metadata(self, *, message_id: str, metadata_json: str) -> bool:
        """Persist structured metadata as a version-neutral, local-only write.

        The task-2364 sibling of ``update_message_usage`` above, with the
        identical contract: metadata-only flush against an already-persisted
        row, no ``version``/``last_modified`` bump (the
        ``messages_sync_update`` trigger watches those, and no sync payload
        can ever carry ``metadata_json``), and entirely optional -- the
        store probes for it and falls back to the content-carrying update
        path when an adapter does not provide it.
        """

    def append_message_exchanges(
        self, *, message_id: str, rows: Sequence[Mapping[str, Any]]
    ) -> bool:
        """Upsert captured provider exchanges for a message (local-only).

        The Conversation Inspector sibling of ``update_message_usage`` --
        each row carries its own ``run_tag``/``seq`` identity, so this is an
        upsert rather than a single-column write. Entirely optional, probed
        the same hasattr+callable way as ``update_message_usage``: a
        persistence adapter that does not implement it simply never
        receives an exchange flush (``ConsoleChatStore._persist_exchanges_
        only`` bails silently rather than falling back to the content path
        -- captures have no content-carrying fallback to ride).
        """

    def get_message_version(self, message_id: str) -> int | None:
        """Return the current positive durable row version, if trustworthy.

        Args:
            message_id: Persisted Chat message identifier.

        Returns:
            The exact positive integer row version, or ``None`` when the row
            cannot provide a trustworthy version fence.
        """

    def get_conversation_version(self, conversation_id: str) -> int | None:
        """Return the current positive durable conversation row version."""

    def get_conversation_speech_preferences(
        self, conversation_id: str
    ) -> ConsoleSpeechPreferences:
        """Return fail-closed reply-speech preferences for one conversation."""

    def update_conversation_speech_preferences(
        self,
        *,
        conversation_id: str,
        preferences: ConsoleSpeechPreferences,
        expected_version: int,
    ) -> bool:
        """Optimistically merge reply-speech preferences into metadata."""

    def update_conversation_system_prompt(
        self,
        *,
        conversation_id: str,
        system_prompt: str | None,
    ) -> bool:
        """Persist a changed system prompt for an already-saved conversation."""

    def update_conversation_roleplay_context(
        self,
        *,
        conversation_id: str,
        user_name_override: str | None,
        character_system_template: str | None,
        character_name_snapshot: str | None,
    ) -> bool:
        """Persist Console-owned roleplay identity context for a conversation.

        Args:
            conversation_id: Durable conversation identifier.
            user_name_override: Optional saved user display-name override.
            character_system_template: Optional saved character prompt template.
            character_name_snapshot: Optional historical character display name.

        Returns:
            True when the roleplay context was persisted.
        """

    def update_conversation_pinned_prefill(
        self,
        *,
        conversation_id: str,
        pinned_prefill: str | None,
    ) -> bool:
        """Set or clear the pinned response prefill on a conversation."""

    def update_conversation_title(
        self,
        *,
        conversation_id: str,
        title: str,
    ) -> bool:
        """Persist a changed title for an already-saved conversation.

        Args:
            conversation_id: Durable Chat conversation identifier.
            title: New conversation title (already validated non-blank).

        Returns:
            True when the update was applied; False when refused (e.g. an
            optimistic-lock version check failed).
        """

    def get_conversation_console_project_context(
        self, *, conversation_id: str
    ) -> str | None:
        """Return versioned local project-context JSON when available."""

    def set_conversation_console_project_context(
        self,
        *,
        conversation_id: str,
        project_context_json: str | None,
    ) -> None:
        """Write local project-context JSON without synchronized metadata."""

    def get_attachments_for_messages(
        self, message_ids: Sequence[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Batch-fetch extra (position >= 1) attachments for messages.

        Optional: not all persistence fakes implement this. Callers should
        probe with ``getattr(persistence, "get_attachments_for_messages", None)``
        before invoking it (see Task 5).
        """

    def append_message_attachment(
        self,
        message_id: str,
        *,
        data: bytes,
        mime_type: str,
        display_name: str = "",
        generation_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Append one new image variant to a message, in place (no rewrite).

        Optional: not all persistence fakes implement this. Callers should
        probe with ``getattr(persistence, "append_message_attachment", None)``
        before invoking it -- the narrow, additive counterpart to
        ``update_message_content(attachments=...)`` used by
        ``ConsoleChatStore.append_generation_variant``.
        """

    def keep_message_attachment(self, message_id: str, position: int) -> None:
        """Promote a stored variant to be the message's canonical image.

        Optional: not all persistence fakes implement this. Callers should
        probe with ``getattr(persistence, "keep_message_attachment", None)``
        before invoking it -- a targeted position swap, used by
        ``ConsoleChatStore.keep_generation_variant`` instead of the
        full-list ``update_message_content(attachments=...)`` rewrite (which
        would NULL any in-memory byte-less variant it re-sends).
        """

    def get_generation_metadata_for_messages(
        self, message_ids: Sequence[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Batch-fetch generation-metadata sidecar rows for messages.

        Optional: not all persistence fakes implement this. Callers should
        probe with
        ``getattr(persistence, "get_generation_metadata_for_messages", None)``
        before invoking it -- feeds
        ``ConsoleChatStore.hydrate_generation_metadata`` at conversation
        load.
        """


class ConsoleChatSyncProducer(Protocol):
    """Sync v2 producer surface used after durable local Chat writes."""

    def enqueue_chat_message(self, **kwargs: Any) -> dict[str, Any]:
        """Enqueue a Chat message into the Sync v2 local outbox."""

    def reconcile_chat_message_intent(self, **kwargs: Any) -> dict[str, Any]:
        """Project an exact committed source intent into durable sync state."""

    def reconcile_chat_message_delete_intent(self, **kwargs: Any) -> dict[str, Any]:
        """Project an exact committed tombstone into durable sync state."""


@dataclass(frozen=True, slots=True)
class ContinuationDurabilityResult:
    """Bounded, private-data-free result for the execution durability barrier."""

    ready: bool
    reason: str


class CapturePolicyStaleError(RuntimeError):
    """A capture-policy write lost its process-local revision race."""


class CapturePurgeStaleError(RuntimeError):
    """A staged capture purge lost its process-local revision race."""


@dataclass(frozen=True, slots=True)
class CapturePolicyState:
    """Process-local capture policy owned by one Console session."""

    next_detail: CaptureDetail | None
    conversation_detail: CaptureDetail | None
    next_revision: int
    policy_revision: int
    capture_revision: int
    save_pending: bool


@dataclass(frozen=True, slots=True)
class StagedCapturePurge:
    """Precomputed live/cache replacements for one Full-capture purge."""

    session_id: str
    conversation_id: str | None
    expected_revision: int
    durable_keys: frozenset[tuple[str, str, int]]
    message_swaps: tuple[tuple[ConsoleChatMessage, tuple["ExchangeCapture", ...]], ...]
    blob_cache: tuple[
        tuple[str, Mapping[tuple[str, int, str], bytes]], ...
    ]
    abandoned_tags: tuple[tuple[str, frozenset[str]], ...]
    capture_revisions: tuple[tuple[ConsoleChatSession, int], ...]
    removed_count: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _invalid_runtime_backend_diagnostic(value: Any) -> str:
    """Return bounded diagnostic context without invoking custom ``repr``."""
    if type(value) is str:
        truncated = value[:120]
        suffix = "..." if len(value) > len(truncated) else ""
        return f"{truncated!r}{suffix}"
    return f"<{type(value).__name__[:120]}>"


@dataclass
class ConsoleChatSession:
    """A native Console chat session."""

    title: str = DEFAULT_CONSOLE_SESSION_TITLE
    workspace_id: str = CONSOLE_GLOBAL_WORKSPACE_ID
    id: str = field(default_factory=lambda: str(uuid4()))
    persisted_conversation_id: str | None = None
    settings: ConsoleSessionSettings | None = None
    #: Exact canonical defaults from which an untouched initial session was
    #: created. ``None`` means the settings have no proven default provenance.
    canonical_settings_baseline: ConsoleSessionSettings | None = field(
        default=None,
        kw_only=True,
    )
    #: Sparse conversation-owned context policy. For an empty unsaved tab this
    #: remains staged only in the session/screen snapshot and is flushed after
    #: the first durable conversation row is created.
    context_policy_overrides: ConsoleContextPolicyOverrides = field(
        default_factory=ConsoleContextPolicyOverrides
    )
    #: Bounded persistence diagnostic for a corrupt/unreadable stored policy.
    context_policy_error: str | None = None
    library_policy_holder: ConsoleLibraryPolicyHolder = field(
        default_factory=_default_library_policy_holder
    )
    #: Restored durable sessions remain fail-closed until off-loop hydration.
    library_policy_hydrated: bool = True
    #: Live-only resolved endpoint/disclosure state. It has no persistence,
    #: sync, import, export, or policy write-through seam.
    library_destination_runtime: ConsoleLibraryDestinationRuntimeState = field(
        default_factory=ConsoleLibraryDestinationRuntimeState
    )
    draft: str = ""
    #: Session-lifetime evidence that the composer has held user-authored text.
    #: Clearing the draft does not make that work safe to overwrite.
    has_user_work: bool = False
    updated_at: str = field(default_factory=_utc_now_iso)
    pending_attachments: list[PendingAttachment] = field(default_factory=list)
    one_shot_prefill: str | None = None
    #: Live opaque identity for the one-shot slot. Every write, including
    #: clearing or re-arming the same text, advances this token.
    one_shot_prefill_revision: int = 0
    capture_detail_override: CaptureDetail | None = None
    next_capture_detail: CaptureDetail | None = None
    next_capture_detail_revision: int = 0
    capture_revision: int = 0
    capture_policy_save_pending: bool = False
    #: RAG retrieval scope (task-9) for a not-yet-persisted session -- see
    #: ``SessionScopeHolder``. ``persist_session_if_needed`` flushes it
    #: through to durable storage exactly once, at first persistence.
    rag_scope_holder: SessionScopeHolder = field(default_factory=SessionScopeHolder)
    #: Durable assistant identity. Authority may be ``None`` for an unscoped
    #: character; such a session still routes as direct character chat but is
    #: not eligible for authority-scoped profile assignment.
    runtime_backend: str = "local"
    assistant_kind: str | None = "generic"
    assistant_id: str | None = "console"
    assistant_authority_id: str | None = None
    #: Local-only numeric compatibility/display projection. Server character
    #: IDs remain opaque in ``assistant_id`` and never populate this field.
    character_id: int | None = None
    character_name: str | None = None
    #: Per-chat human label, independently persisted in conversation metadata.
    user_display_name_override: str | None = None
    #: Trusted character system source; materialized into ``settings.system_prompt``.
    character_system_template: str | None = None
    speech_preferences: ConsoleSpeechPreferences = field(
        default_factory=ConsoleSpeechPreferences
    )
    #: Monotonic identity projection fence for labels and trusted templates.
    identity_revision: int = 0
    project_instruction_state: ProjectInstructionControlState = field(
        default_factory=ProjectInstructionControlState.legacy_disabled
    )
    #: Temporary conversation (spec 2026-07-31): this session is never written
    #: to local storage. Enforced in exactly one place --
    #: ``persist_session_if_needed`` refuses to mint a
    #: ``persisted_conversation_id`` -- so every durable write downstream
    #: no-ops along the branch it already takes with no persistence adapter.
    #: A write site that forgets about this flag therefore fails toward NOT
    #: writing, which is the whole reason the guard lives at the id and not
    #: at the 43 sites that consult ``self.persistence``.
    ephemeral: bool = False

    def local_character_id(self) -> int | None:
        """Return the exact validated local character projection, if any."""
        if (
            self.runtime_backend != "local"
            or self.assistant_kind != "character"
            or type(self.character_id) is not int
            or self.character_id < 1
            or self.assistant_id != str(self.character_id)
        ):
            return None
        return self.character_id

    def character_ref(self) -> CharacterRef | None:
        """Return the complete authority-scoped character identity, if any."""
        if self.assistant_kind != "character":
            return None
        if type(self.assistant_authority_id) is not str:
            return None
        if type(self.assistant_id) is not str or not self.assistant_id:
            return None

        if self.runtime_backend == "local":
            if self.local_character_id() is None:
                return None
        elif self.runtime_backend == "server":
            if self.character_id is not None:
                return None
        else:
            return None

        try:
            return CharacterRef(
                source=self.runtime_backend,
                authority_id=self.assistant_authority_id,
                character_id=self.assistant_id,
            )
        except ProfileValidationError:
            return None

    #: Session-lifetime stable task store for the local task tools. Durable
    #: conversation resume deliberately starts with a fresh empty store.
    todo_store: SessionTodoStore = field(default_factory=SessionTodoStore)


def is_untouched_default_session(
    session: ConsoleChatSession,
    messages: Iterable[object],
    draft: str,
    staged_attachments: Iterable[object],
) -> bool:
    """Return whether visible session state proves a default tab is untouched."""

    if not isinstance(session, ConsoleChatSession):
        return False
    if type(draft) is not str or draft:
        return False
    sentinel = object()
    try:
        if next(iter(messages), sentinel) is not sentinel:
            return False
        if next(iter(staged_attachments), sentinel) is not sentinel:
            return False
    except (TypeError, RuntimeError):
        return False
    baseline = session.canonical_settings_baseline
    if baseline is None or session.settings != baseline:
        return False
    return not (
        session.title != DEFAULT_CONSOLE_SESSION_TITLE
        or session.persisted_conversation_id is not None
        or session.draft != ""
        or session.has_user_work
        or session.pending_attachments
        or session.one_shot_prefill is not None
        or session.rag_scope_holder.scope is not None
        or not session.context_policy_overrides.is_empty
        or session.context_policy_error is not None
        or session.library_policy_holder.explicitly_staged
        or session.library_policy_holder.save_pending
        or session.runtime_backend != "local"
        or session.assistant_kind != "generic"
        or session.assistant_id != "console"
        or session.assistant_authority_id is not None
        or session.character_id is not None
        or session.character_name is not None
        or session.user_display_name_override is not None
        or session.character_system_template is not None
        or session.speech_preferences != ConsoleSpeechPreferences()
        or session.identity_revision != 0
        or session.ephemeral
        or session.todo_store.list_after(None)
    )


class ConsoleChatStore:
    """Manage native Console sessions and messages before UI integration."""

    DURABLE_TOMBSTONE_CAP = 128

    def __init__(
        self,
        *,
        persistence: ConsoleChatPersistence | None = None,
        workspace_context: ConsoleWorkspaceContext | None = None,
        sync_v2_chat_producer: ConsoleChatSyncProducer | None = None,
        sync_v2_server_profile_id: str | None = None,
        sync_v2_authenticated_principal_id: str | None = None,
        sync_v2_workspace_scope: str | None = None,
        on_scope_flushed: Callable[[str, "RagScope | None"], None] | None = None,
        library_policy_defaults: ConsoleLibraryPolicyDefaults | None = None,
        library_policy_defaults_provider: (
            Callable[[], ConsoleLibraryPolicyDefaults] | None
        ) = None,
        library_policy_coordinator: ConsoleLibraryPolicyCoordinator | None = None,
    ) -> None:
        """Initialize the Console chat store.

        Args:
            persistence: Optional durable Chat persistence adapter.
            workspace_context: Current workspace and staged-source policy context.
            sync_v2_chat_producer: Optional Sync v2 producer called after durable,
                complete local Chat message writes.
            sync_v2_server_profile_id: Optional server profile scope for Chat outbox
                entries. If missing, Sync v2 enqueue is disabled.
            sync_v2_authenticated_principal_id: Optional authenticated principal scope
                for Chat outbox entries.
            sync_v2_workspace_scope: Optional workspace scope for Chat outbox entries.
            on_scope_flushed: Optional callback invoked with
                ``(conversation_id, scope)`` immediately after
                ``persist_session_if_needed`` successfully flushes a
                session-held RAG retrieval scope (``SessionScopeHolder``)
                through to durable storage at first persistence (task-9
                review finding 1). This is the ONLY moment a session
                transitions from "scope held in memory" to "scope persisted
                under a new conversation id" without going through any of
                the UI's other read triggers (resume, modal-open,
                after-save) -- callers that keep a display-side cache keyed
                by conversation id (e.g. the Console Inspector's retrieval-
                scope row) use this hook to stay in sync instead of reading
                stale/absent cache state. Never called when nothing was
                held, or when the flush itself raised.
        """
        self.persistence = persistence
        self.workspace_context = workspace_context or ConsoleWorkspaceContext()
        self.sync_v2_chat_producer = sync_v2_chat_producer
        self.sync_v2_server_profile_id = sync_v2_server_profile_id
        self.sync_v2_authenticated_principal_id = sync_v2_authenticated_principal_id
        self.sync_v2_workspace_scope = sync_v2_workspace_scope
        self.on_scope_flushed = on_scope_flushed
        self._library_policy_defaults = library_policy_defaults or (
            ConsoleLibraryPolicyDefaults(
                auto_retrieve=ConsoleAutoRetrieve.NEVER,
                assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            )
        )
        self._library_policy_defaults_provider = library_policy_defaults_provider
        repository = getattr(
            persistence,
            "console_library_policy_repository",
            None,
        )
        self.library_policy_coordinator = library_policy_coordinator
        if self.library_policy_coordinator is None and repository is not None:
            self.library_policy_coordinator = ConsoleLibraryPolicyCoordinator(
                repository
            )
        self.active_session_id: str | None = None
        self._active_session_epoch = 0
        self._speech_preference_epoch_sequence = 0
        self._sessions: dict[str, ConsoleChatSession] = {}
        self._capture_policy_revision = 0
        self._capture_policy_lock = threading.RLock()
        self._capture_policy_mutation: object | None = None
        capture_policy_db = getattr(persistence, "db", None)
        self.capture_policy_repository = (
            ConsoleCapturePolicyRepository(capture_policy_db)
            if capture_policy_db is not None
            else None
        )
        # Task 13: one app-lifetime, process-memory preparation owner per
        # session.  This state deliberately belongs to the store rather than
        # a mounted Console screen so navigation cannot lose or duplicate a
        # pre-dispatch decision.  All mutations use exact preparation/state
        # compare-and-set under one lock; the immutable values are safe to
        # return directly.
        self._preparation_lock = threading.RLock()
        self._preparations_by_session: dict[str, ConsoleTurnPreparation] = {}
        self._preparations_by_id: dict[str, ConsoleTurnPreparation] = {}
        self._durable_identity_by_preparation: dict[
            str, ConsoleStagedConversationIdentity
        ] = {}
        self._durable_owner_ids_by_preparation: dict[
            str, _ConsoleStagedDurableOwnerIds
        ] = {}
        self._durable_commit_by_preparation: dict[str, ConsoleDurableTurnCommit] = {}
        self._durable_effects_by_preparation: dict[
            str, ConsoleDurablePostcommitEffects
        ] = {}
        self._durable_effects_in_flight: set[tuple[str, str]] = set()
        self._durable_commit_in_flight: dict[str, _ConsoleDurableCommitReservation] = {}
        self._durable_fingerprint_by_preparation: dict[
            str, ConsoleDurableAcceptanceFingerprint
        ] = {}
        #: Preparations whose postcommit sequence has begun and has not yet
        #: been released. Their tombstones are the ONLY proof that an
        #: in-flight effect's preparation was retired rather than mutated,
        #: so FIFO eviction must not reclaim them first (TASK-22587).
        self._durable_active_postcommit: set[str] = set()
        self._durable_tombstones: OrderedDict[str, _ConsoleDurableTombstone] = (
            OrderedDict()
        )
        # Task 15: the app-runtime owner for both reconciled durable recovery
        # and the no-SQL ephemeral analogue.  Mounted screens only project it.
        self._dispatch_recoveries_by_session: dict[
            str, ConsoleDispatchRecoveryState
        ] = {}
        self._dispatch_recovery_message_baselines: dict[str, ConsoleChatMessage] = {}
        self._dispatch_recovery_queue_hydration_pending: set[str] = set()
        #: Derived VIEW = the current active path only (root -> active leaf).
        #: Written ONLY by ``_recompute_active_path`` (single-writer invariant);
        #: every other reader/writer of the tree goes through the maps below.
        self._messages_by_session: dict[str, list[ConsoleChatMessage]] = {}
        #: TASK-1842: display-only TOOL markers, each paired with the id of the
        #: node it followed. Kept OUTSIDE the tree on purpose -- a marker that
        #: became a node would corrupt the parent chain (see the invariant in
        #: `append_message`) -- but kept somewhere durable so
        #: `_recompute_active_path`, the single writer of the view above, can
        #: splice them back instead of erasing an agent's whole tool trace on
        #: the user's next message.
        self._tool_markers_by_session: dict[
            str, list[tuple[str | None, ConsoleChatMessage]]
        ] = {}
        self._message_session_index: dict[str, str] = {}
        #: Full conversation tree -- ALL branches, on- and off-path. ``_nodes``
        #: maps a native id to the LIVE ``ConsoleChatMessage`` (never a copy --
        #: streaming mutates content in place and the derived view must observe
        #: it). ``_children`` maps a native parent id (``None`` for roots) to the
        #: ordered child native ids. ``_native_parent`` maps a native id to its
        #: native parent id (``None`` for a root). Distinct from a message's
        #: ``parent_message_id`` field, which is the *persisted* parent id.
        self._nodes_by_session: dict[str, dict[str, ConsoleChatMessage]] = {}
        self._children_by_parent: dict[str, dict[str | None, list[str]]] = {}
        self._native_parent_by_message: dict[str, str | None] = {}
        self._active_leaf_by_session: dict[str, str | None] = {}
        #: Console `/rewind` "summarize up to here" (SP2): per-session
        #: ``(summary, boundary_native_id)`` pair. Local-only, mirrors
        #: ``_active_leaf_by_session`` -- a parallel dict, not tree state, so
        #: it is untouched by tree mutations (create/delete/sibling). ``(None,
        #: None)`` = no summary. Write-through is ``_persist_context_summary``.
        self._context_summary_by_session: dict[str, tuple[str | None, str | None]] = {}
        self._deferred_project_instruction_state_session_ids: set[str] = set()
        self._unresolved_promotion_operations: dict[str, str] = {}
        self._pending_workspace_projections: dict[str, str] = {}
        self._pending_persistence_message_ids: set[str] = set()
        self._terminal_citation_finalizers: dict[str, TerminalCitationFinalizer] = {}
        self._provisional_terminal_selection_ids: set[str] = set()
        self._terminal_persistence_deferred_ids: set[str] = set()
        self._stream_chunks_by_message: dict[str, list[str]] = {}
        self._stream_materialized_counts: dict[str, int] = {}
        self._character_emote_captures: dict[str, _CharacterEmoteCapture] = {}
        self._character_emote_feed_by_session: dict[
            str, deque[CharacterEmoteLiveEvent]
        ] = {}
        self._character_emote_sequence = 0
        self._sync_v2_message_versions: dict[str, str] = {}
        self._roleplay_system_projection_candidates: dict[
            str, tuple[str | None, ...]
        ] = {}
        self._roleplay_message_projection_candidates: dict[str, tuple[str, ...]] = {}
        self._variant_stream_bases: dict[str, _VariantStreamBase] = {}
        # Messages whose CURRENT content was RESTORED from a pre-regenerate
        # base by a stopped/failed terminal mark. Their content belongs to
        # the ORIGINAL generation, so a late usage attach arriving from the
        # abandoned run (the Stop path finalizes the message first and only
        # then cancels the stream task, whose CancelledError handler
        # attaches) must not land on -- let alone persist over -- the
        # original's own record. Cleared the moment a new generation starts
        # on the message (`begin_variant_stream`/`prepare_message_retry`).
        self._variant_restored_message_ids: set[str] = set()
        # Run tags whose captures were attached AFTER a variant-restore --
        # the traffic really happened, so (unlike usage) it is kept rather
        # than dropped, but flagged ``abandoned`` on the durable row so a
        # viewer can tell it apart from the answer actually shown. Keyed by
        # message id; a run_tag once marked abandoned here stays abandoned
        # for every later flush of that message (see
        # ``attach_message_exchanges``/``_persist_exchanges_only``).
        self._abandoned_exchange_run_tags: dict[str, set[str]] = {}
        # Qodo PR #1883 finding 4: memoizes ``capture_to_blob``'s
        # zlib-compressed JSON per (message, run_tag, seq, status) so
        # ``_persist_exchanges_only`` -- called on EVERY flush of a
        # message with exchanges, e.g. once per tool call in a long agent
        # turn -- does not re-compress every still-unchanged capture on
        # every call. Keyed by message id (outer) then (run_tag, seq,
        # status) (inner) so one message's blobs can never serve another's
        # lookup. ``status`` is part of the inner key because a "stopped"
        # snapshot can legitimately be superseded by a later non-"stopped"
        # capture for the same (run_tag, seq) (see
        # ``attach_message_exchanges``'s merge rule) -- that status change
        # naturally misses this cache and recompresses the new bytes.
        # ``_persist_exchanges_only`` prunes each message's inner dict down
        # to exactly its current capture keys on every flush, so a
        # superseded status's stale blob does not linger past its own
        # message's next persist. M2 (softened -- this used to read as an
        # unqualified "cannot grow past what is actually live" bound, which
        # was not true of every path a message can disappear by):
        # ``delete_message`` and session-close both drop a message's
        # entire entry outright when the message itself goes away, and
        # ``restore_state`` clears this cache wholesale on a full state
        # replacement (session switch / restart replay) -- but only those
        # three sites do the dropping; an in-memory message id that
        # disappears some OTHER way is not itself proof this cache's entry
        # for it goes away too.
        self._exchange_blob_cache: dict[str, dict[tuple[str, int, str], bytes]] = {}
        self._capture_quiescence_lock = threading.RLock()
        self._capture_quiescent_sessions: set[str] = set()
        # Ephemeral fence for issued speech snapshots. It deliberately lives
        # outside ConsoleChatMessage so it is neither persisted nor restored.
        self._message_speech_revisions: dict[str, int] = {}
        # Content-free fence that advances only for live successful
        # completions. It distinguishes duplicate callback delivery from a
        # later regeneration of the same message without retaining text.
        self._message_completion_generations: dict[str, int] = {}
        self._message_completion_epoch = 0
        # Cost-ticker PR3: per-session monotonic counter of payload-affecting
        # mutations, so the cost chip knows when its cache-break fingerprint
        # needs recomputing. Process-local, like the speech revisions above.
        self._payload_revisions: dict[str, int] = {}
        # Prompt-queue context safety: a narrower process-local token than
        # `_payload_revisions`. Ordinary linear turn growth and streaming do
        # not move it; out-of-band changes to the effective active provider
        # context do. It is intentionally absent from persistence/snapshots.
        self._conversation_context_epochs: dict[str, int] = {}
        # A failed assistant row is excluded from ordinary provider payloads.
        # Retrying it reuses that row in place, so the eventual complete or
        # stopped terminal must advance the context epoch when the row becomes
        # provider-visible history. A repeat failure stays excluded and stable.
        self._failed_retry_message_ids: set[str] = set()
        # Process-local observers for the first LIVE transition of a message
        # into the complete state. Restored/already-complete rows never pass
        # through the publisher, so hydration cannot replay speech.
        self._message_completed_subscribers: dict[
            int, Callable[[tuple[str, str]], None]
        ] = {}
        self._next_message_completed_subscriber_id = 1
        self._speech_preference_epochs: dict[str, int] = {}

        # Trajectory sidecar (schema v38) capture state. LOCAL-ONLY: the
        # ``message_trajectory_metadata`` table is never synced. Timing is
        # armed by the controller (step start / completion) and stamped here
        # (first token, at the chunk seam); rows are written through the
        # persistence adapter in ONE batched upsert per turn. Every write is
        # best-effort: a sidecar failure must never fail the turn.
        self._trajectory_lock = threading.Lock()
        self._trajectory_timing: dict[str, dict[str, Any]] = {}
        self._trajectory_written_ids: set[str] = set()
        self._session_turn_ids: dict[str, str] = {}
        # TOOL markers appended while their parent assistant row is still
        # streaming (no durable id yet): payload captured at marker-append
        # time, flushed -- remapped to the parent's persisted id -- when the
        # parent message persists. Keyed by the parent's NATIVE message id.
        self._pending_trajectory_tool_rows: dict[str, list[dict[str, Any]]] = {}
        self._pending_trajectory_event_rows: dict[str, list[dict[str, Any]]] = {}
        self._trajectory_capture_failure_keys: set[str] = set()
        self._trajectory_capture_failure_hydrated: set[str] = set()

    def subscribe_message_completed(
        self,
        callback: Callable[[tuple[str, str]], None],
    ) -> Callable[[], None]:
        """Observe successful live completion tokens until unsubscribe.

        The callback receives only an immutable ``(session_id, message_id)``
        token. Subscriber failures are isolated from message finalization.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        subscriber_id = self._next_message_completed_subscriber_id
        self._next_message_completed_subscriber_id += 1
        self._message_completed_subscribers[subscriber_id] = callback
        subscribed = True

        def unsubscribe() -> None:
            nonlocal subscribed
            if not subscribed:
                return
            subscribed = False
            self._message_completed_subscribers.pop(subscriber_id, None)

        return unsubscribe

    def _publish_message_completed(self, session_id: str, message_id: str) -> None:
        """Publish one validated live terminal transition without message content."""
        if self._message_session_index.get(message_id) != session_id:
            return
        token = (session_id, message_id)
        for callback in tuple(self._message_completed_subscribers.values()):
            try:
                callback(token)
            except Exception:
                logger.warning("Console completion subscriber failed")

    def _record_message_completed(self, session_id: str, message_id: str) -> None:
        self._message_completion_epoch += 1
        self._message_completion_generations[message_id] = (
            self._message_completion_epoch
        )
        self._settle_message_library_destination(session_id, message_id)
        self._publish_message_completed(session_id, message_id)

    def message_completion_generation(self, message_id: str) -> int:
        """Return the process-local generation of a live successful completion."""
        self._message_or_raise(message_id)
        return self._message_completion_generations[message_id]

    def ensure_session(
        self,
        *,
        title: str = DEFAULT_CONSOLE_SESSION_TITLE,
        workspace_id: str | None = None,
        settings: ConsoleSessionSettings | None = None,
        canonical_settings_baseline: ConsoleSessionSettings | None = None,
    ) -> ConsoleChatSession:
        """Return the active session, creating one when needed."""
        if self.active_session_id is not None:
            return self._sessions[self.active_session_id]
        return self.create_session(
            title=title,
            workspace_id=workspace_id,
            settings=settings,
            canonical_settings_baseline=canonical_settings_baseline,
        )

    def create_session(
        self,
        *,
        session_id: str | None = None,
        title: str = DEFAULT_CONSOLE_SESSION_TITLE,
        workspace_id: str | None = None,
        settings: ConsoleSessionSettings | None = None,
        canonical_settings_baseline: ConsoleSessionSettings | None = None,
        runtime_backend: str = "local",
        assistant_kind: str | None = "generic",
        assistant_id: str | None = "console",
        assistant_authority_id: str | None = None,
        character_id: int | None = None,
        character_name: str | None = None,
        ephemeral: bool = False,
        activate: bool = True,
        project_instruction_state: ProjectInstructionControlState | None = None,
    ) -> ConsoleChatSession:
        """Create and activate a new native Console session.

        Args:
            session_id: Optional validated identity reserved by a typed handoff.
            canonical_settings_baseline: Exact canonical defaults that equal
                ``settings``. Omit when the settings have no proven provenance.
            ephemeral: When True the session is temporary -- never written to
                local storage until ``promote_ephemeral_session`` clears the
                flag.
            activate: Whether to make the new session active immediately.
        """
        if session_id is not None:
            if (
                type(session_id) is not str
                or not session_id
                or session_id != session_id.strip()
                or len(session_id) > 256
            ):
                raise ValueError("session id is invalid")
            if session_id in self._sessions:
                raise ValueError("session id already exists")
        if canonical_settings_baseline is not None and (
            not isinstance(
                canonical_settings_baseline,
                ConsoleSessionSettings,
            )
            or canonical_settings_baseline != settings
        ):
            raise ValueError("canonical baseline must equal the session settings.")
        defaults = (
            self._library_policy_defaults_provider()
            if self._library_policy_defaults_provider is not None
            else self._library_policy_defaults
        )
        if not isinstance(defaults, ConsoleLibraryPolicyDefaults):
            raise TypeError(
                "library policy defaults provider must return "
                "ConsoleLibraryPolicyDefaults"
            )
        session = ConsoleChatSession(
            id=session_id or str(uuid4()),
            title=title,
            workspace_id=workspace_id or self.workspace_context.active_workspace_id,
            settings=settings,
            canonical_settings_baseline=canonical_settings_baseline,
            runtime_backend=runtime_backend,
            assistant_kind=assistant_kind,
            assistant_id=assistant_id,
            assistant_authority_id=assistant_authority_id,
            character_id=character_id,
            character_name=character_name,
            ephemeral=ephemeral,
            project_instruction_state=(
                project_instruction_state
                if project_instruction_state is not None
                else ProjectInstructionControlState.new_session()
            ),
            library_policy_holder=ConsoleLibraryPolicyHolder(
                ConsoleLibraryPolicySnapshot(
                    auto_retrieve=defaults.auto_retrieve,
                    assistant_access=defaults.assistant_access,
                    policy_revision=None,
                    source="temporary" if ephemeral else "new_session",
                )
            ),
        )
        self._sessions[session.id] = session
        self._messages_by_session[session.id] = []
        self._nodes_by_session[session.id] = {}
        self._children_by_parent[session.id] = {}
        self._active_leaf_by_session[session.id] = None
        self._context_summary_by_session[session.id] = (None, None)
        self._conversation_context_epochs[session.id] = 0
        if self.library_policy_coordinator is not None:
            self.library_policy_coordinator.register_holder(
                session.id,
                None,
                session.library_policy_holder,
            )
        if activate:
            self._activate_session(session.id)
        return session

    def _activate_session(self, session_id: str | None) -> None:
        """Publish one activation transition behind a monotonic process fence."""
        self.active_session_id = session_id
        self._active_session_epoch += 1

    def active_session_epoch(self) -> int:
        """Return the monotonic generation of the current activation state."""
        return self._active_session_epoch

    def is_pristine_session(
        self,
        session_id: str,
        *,
        expected_settings: ConsoleSessionSettings,
    ) -> bool:
        """Return whether a session is the untouched initial Console tab."""
        session = self._sessions.get(session_id)
        if session is None or not isinstance(expected_settings, ConsoleSessionSettings):
            return False
        if session.canonical_settings_baseline != expected_settings:
            return False
        if not is_untouched_default_session(
            session,
            self._messages_by_session.get(session_id, ()),
            session.draft,
            session.pending_attachments,
        ):
            return False
        # Every real message and display-only tool marker is assigned to its
        # session through the full tree and/or `_message_session_index` at its
        # registration boundary. Any such ownership is therefore a complete
        # strict disqualifier, even when the active-path message list is empty.
        # Per-message cache entries without either ownership source are not
        # attributable to this session and must not be guessed from key text.
        owned_message_state = (
            self._messages_by_session.get(session_id)
            or self._nodes_by_session.get(session_id)
            or self._children_by_parent.get(session_id)
            or self._active_leaf_by_session.get(session_id) is not None
            or any(
                owner == session_id for owner in self._message_session_index.values()
            )
            or bool(self._tool_markers_by_session.get(session_id))
        )
        session_live_state = (
            self._context_summary_by_session.get(session_id) != (None, None)
            or bool(self._roleplay_system_projection_candidates.get(session_id))
            or self._conversation_context_epochs.get(session_id) != 0
        )
        return not (owned_message_state or session_live_state)

    def repurpose_pristine_session(
        self,
        session_id: str,
        *,
        canonical_settings: ConsoleSessionSettings,
        settings: ConsoleSessionSettings,
        trusted_system_prompt: str,
        title: str,
        runtime_backend: str,
        assistant_kind: str,
        assistant_id: str,
        assistant_authority_id: str | None,
        character_id: int | None,
        character_name: str,
    ) -> ConsoleChatSession:
        """Atomically replace an untouched initial tab with roleplay identity."""
        if not isinstance(canonical_settings, ConsoleSessionSettings):
            raise TypeError("canonical_settings must be ConsoleSessionSettings.")
        if not isinstance(settings, ConsoleSessionSettings):
            raise TypeError("settings must be ConsoleSessionSettings.")
        if type(trusted_system_prompt) is not str or not trusted_system_prompt.strip():
            raise ValueError("Trusted roleplay system prompt must be non-empty text.")
        if runtime_backend not in {"local", "server"}:
            raise ValueError("Roleplay runtime backend must be local or server.")
        if assistant_kind != "character":
            raise ValueError("Repurposed sessions require character identity.")
        if type(assistant_id) is not str or not assistant_id:
            raise ValueError("Roleplay assistant id must be non-empty text.")
        if assistant_authority_id is not None and (
            type(assistant_authority_id) is not str or not assistant_authority_id
        ):
            raise ValueError("Roleplay authority id must be non-empty text or None.")
        if type(character_name) is not str or not character_name.strip():
            raise ValueError("Roleplay character name must be non-empty text.")
        if title != f"Chat with {character_name}":
            raise ValueError("Roleplay title does not match the character identity.")
        expected_roleplay_settings = replace(
            canonical_settings,
            system_prompt=trusted_system_prompt,
            character_label=character_name,
        )
        if settings != expected_roleplay_settings:
            raise ValueError("Roleplay settings contain noncanonical changes.")
        if runtime_backend == "local":
            if (
                type(character_id) is not int
                or character_id < 1
                or assistant_id != str(character_id)
            ):
                raise ValueError("Local roleplay identity is inconsistent.")
        elif character_id is not None:
            raise ValueError("Server roleplay identity cannot carry a local id.")

        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("Session is no longer pristine.")
        proposed_updated_at = _utc_now_iso()
        proposed_identity_revision = session.identity_revision + 1
        proposed_payload_revision = self._payload_revisions.get(session_id, 0) + 1
        if not self.is_pristine_session(
            session_id,
            expected_settings=canonical_settings,
        ):
            raise ValueError("Session is no longer pristine.")

        # All validation, derived values, and stale-eligibility checks are
        # complete. ConsoleChatSession is a plain dataclass with no property
        # setters, so these bounded assignments cannot raise application-level
        # exceptions and preserve the live object references held by the UI.
        session.title = title
        session.settings = settings
        session.canonical_settings_baseline = None
        session.runtime_backend = runtime_backend
        session.assistant_kind = assistant_kind
        session.assistant_id = assistant_id
        session.assistant_authority_id = assistant_authority_id
        session.character_id = character_id
        session.character_name = character_name
        session.updated_at = proposed_updated_at
        session.identity_revision = proposed_identity_revision
        self._payload_revisions[session_id] = proposed_payload_revision
        return session

    def refresh_pristine_session_settings(
        self,
        session_id: str,
        *,
        prior_canonical_settings: ConsoleSessionSettings,
        current_canonical_settings: ConsoleSessionSettings,
    ) -> ConsoleChatSession:
        """Atomically refresh proven canonical defaults on an untouched tab."""
        if not isinstance(
            prior_canonical_settings, ConsoleSessionSettings
        ) or not isinstance(current_canonical_settings, ConsoleSessionSettings):
            raise TypeError("Canonical settings provenance is required.")
        if not all(
            settings.source == "derived"
            and settings.system_prompt is None
            and settings.character_label == ""
            and settings.pinned_prefill is None
            for settings in (
                prior_canonical_settings,
                current_canonical_settings,
            )
        ):
            raise ValueError("Canonical settings must be unmodified derived defaults.")
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("Session is no longer pristine.")
        if session.canonical_settings_baseline != prior_canonical_settings:
            raise ValueError("Session settings lack the expected canonical provenance.")
        proposed_updated_at = _utc_now_iso()
        if not self.is_pristine_session(
            session_id,
            expected_settings=prior_canonical_settings,
        ):
            raise ValueError("Session is no longer pristine.")

        # Validation and stale-eligibility checks are complete. These are plain
        # dataclass assignments with no property setters, so the bounded commit
        # cannot raise application-level exceptions and preserves held references.
        session.settings = current_canonical_settings
        session.canonical_settings_baseline = current_canonical_settings
        session.updated_at = proposed_updated_at
        return session

    def rollback_pristine_session_refresh(
        self,
        session_id: str,
        *,
        expected_current_settings: ConsoleSessionSettings,
        prior_settings: ConsoleSessionSettings,
        prior_canonical_settings: ConsoleSessionSettings,
        prior_updated_at: str,
    ) -> bool:
        """Restore an exact first-chat refresh only while it remains pristine."""

        session = self._sessions.get(session_id)
        if (
            session is None
            or session.settings != expected_current_settings
            or session.canonical_settings_baseline != expected_current_settings
            or not self.is_pristine_session(
                session_id,
                expected_settings=expected_current_settings,
            )
        ):
            return False
        session.settings = prior_settings
        session.canonical_settings_baseline = prior_canonical_settings
        session.updated_at = prior_updated_at
        return True

    def rollback_created_pristine_session(
        self,
        session_id: str,
        *,
        expected_session: ConsoleChatSession,
        expected_settings: ConsoleSessionSettings,
        prior_active_session_id: str | None,
    ) -> bool:
        """Remove an exact newly-created target without touching claimed work."""

        session = self._sessions.get(session_id)
        if session is not expected_session or not self.is_pristine_session(
            session_id,
            expected_settings=expected_settings,
        ):
            return False
        self._purge_session_runtime_state(session_id)
        if self.active_session_id == session_id:
            self._activate_session(
                prior_active_session_id
                if prior_active_session_id in self._sessions
                else None
            )
        return True

    def rollback_restored_session(
        self,
        session_id: str,
        *,
        expected_session: ConsoleChatSession,
        prior_active_session_id: str | None,
    ) -> bool:
        """Remove only the exact runtime session created by a failed restore.

        Args:
            session_id: Runtime session identifier to remove.
            expected_session: Exact restored session instance that still owns cleanup.
            prior_active_session_id: Session to reactivate when it still exists.

        Returns:
            True when the exact restored session was removed; False when ownership
            had changed and no cleanup was performed.
        """

        if self._sessions.get(session_id) is not expected_session:
            return False
        self._purge_session_runtime_state(session_id)
        if self.active_session_id == session_id:
            self._activate_session(
                prior_active_session_id
                if prior_active_session_id in self._sessions
                else None
            )
        return True

    def restore_persisted_session(
        self,
        *,
        title: str,
        workspace_id: str | None,
        persisted_conversation_id: str,
        all_nodes: Iterable[ConsoleChatMessage],
        active_leaf_persisted_id: str | None = None,
        settings: ConsoleSessionSettings | None = None,
        runtime_backend: str = "local",
        assistant_kind: str | None = "generic",
        assistant_id: str | None = "console",
        assistant_authority_id: str | None = None,
        character_id: int | None = None,
        character_name: str | None = None,
        ephemeral: bool = False,
        remote_active: bool = False,
        activate: bool = True,
    ) -> ConsoleChatSession:
        """Create and activate a native session from persisted conversation data.

        Task 8: a restored conversation arrives as the WHOLE persisted tree --
        every branch, on- and off-path -- so off-path siblings are navigable
        (swipe) immediately after resume. ``all_nodes`` is the flattened node
        set (pre-order, every node), each carrying its own
        ``persisted_message_id`` and its persisted ``parent_message_id``; the
        full in-memory tree is rebuilt from those links and the active-path
        VIEW is derived from ``active_leaf_persisted_id`` (falling back to the
        most-recent-child leaf, repairing the durable pointer, when the pointer
        is missing or dangling).

        task-558: also hydrates every restored node's ``generation_metadata``
        from the ``message_generation_metadata`` sidecar table, via one
        batched ``persistence.get_generation_metadata_for_messages`` call
        feeding ``hydrate_generation_metadata`` (see
        ``_hydrate_generation_metadata_from_persistence``) -- callers do not
        need to drive that seam themselves. Silently skipped (nodes stay
        metadata-only) when ``persistence`` is ``None`` or doesn't implement
        the batch-fetch method.

        Args:
            title: Display title for the restored Console session.
            workspace_id: Workspace scope recorded on the persisted conversation,
                or ``None`` to use the current store workspace context.
            persisted_conversation_id: Durable Chat conversation identifier.
            all_nodes: Every native Console node reconstructed from the
                persisted conversation tree (all branches), each carrying its
                ``persisted_message_id`` and persisted ``parent_message_id``.
            active_leaf_persisted_id: Persisted id of the stored active-leaf
                pointer, or ``None``. Selects which branch is the active-path
                view; ``None``/missing/dangling falls back to the most-recent
                leaf and repairs the durable pointer.
            settings: Optional provider/model settings snapshot for the session.

        Returns:
            The newly created and activated Console session.
        """
        # A restored session comes FROM durable storage, so it is by
        # definition not temporary. Refuse rather than silently produce a
        # session that is both temporary and persisted -- the one state the
        # gate's invariant does not allow.
        if ephemeral:
            raise ValueError(
                "Cannot restore a persisted session as temporary: a temporary "
                "session has no persisted conversation."
            )
        project_instruction_state = ProjectInstructionControlState.legacy_disabled()
        getter = getattr(
            self.persistence, "get_conversation_console_project_context", None
        )
        if callable(getter):
            try:
                raw_project_context = getter(
                    conversation_id=str(persisted_conversation_id)
                )
            except Exception:
                logger.warning("project_instruction_state_read_failed")
            else:
                project_instruction_state = decode_project_context_json(
                    raw_project_context
                )
        prior_active_session_id = self.active_session_id
        session = self.create_session(
            title=title,
            workspace_id=workspace_id,
            settings=settings,
            runtime_backend=runtime_backend,
            assistant_kind=assistant_kind,
            assistant_id=assistant_id,
            assistant_authority_id=assistant_authority_id,
            character_id=character_id,
            character_name=character_name,
            project_instruction_state=project_instruction_state,
            activate=activate,
        )
        try:
            session.persisted_conversation_id = str(persisted_conversation_id)
            self.hydrate_session_capture_policy(session.id)
            self._hydrate_dispatch_recovery(
                session.id,
                str(persisted_conversation_id),
            )
            session.library_policy_hydrated = False
            coordinator = self.library_policy_coordinator
            if coordinator is not None:
                session.library_policy_holder.snapshot = normalize_policy_read(
                    None
                ).snapshot
                session.library_policy_holder.explicitly_staged = False
                coordinator.register_holder(
                    session.id,
                    session.persisted_conversation_id,
                    session.library_policy_holder,
                )
            if session.workspace_id != CONSOLE_GLOBAL_WORKSPACE_ID:
                self._pending_workspace_projections[session.id] = (
                    session.persisted_conversation_id
                )
            self._restore_speech_preferences(session)
            self._resolve_context_policy_on_resume(session.id)
            restored_nodes = self._hydrate_provider_continuations_from_persistence(
                session.id,
                persisted_conversation_id,
                list(all_nodes),
                remote_active=remote_active,
            )
            self._ingest_full_tree(
                session.id,
                restored_nodes,
                active_leaf_persisted_id=active_leaf_persisted_id,
            )
            self._normalize_restored_provider_continuation(
                session.id, str(persisted_conversation_id)
            )
            self._reconcile_restored_chat_sync_intents(
                session.id, str(persisted_conversation_id)
            )
            self._hydrate_generation_metadata_from_persistence(session.id)
            self._bump_payload_revision(session.id)
            return session
        except BaseException:
            self.rollback_restored_session(
                session.id,
                expected_session=session,
                prior_active_session_id=prior_active_session_id,
            )
            raise

    def _hydrate_dispatch_recovery(
        self,
        session_id: str,
        conversation_id: str,
    ) -> ConsoleDispatchRecoveryState | None:
        """Reconcile device-local ownership before any restored queue can wake."""

        repository = getattr(
            self.persistence,
            "console_dispatch_repository",
            None,
        )
        reconcile = getattr(repository, "reconcile_for_session", None)
        if not callable(reconcile):
            self._dispatch_recoveries_by_session.pop(session_id, None)
            self._dispatch_recovery_message_baselines.pop(session_id, None)
            self._dispatch_recovery_queue_hydration_pending.discard(session_id)
            return None
        try:
            recovery = reconcile(conversation_id)
        except Exception:
            logger.warning("console_dispatch_recovery_hydration_failed")
            recovery = ConsoleDispatchRecoveryState(
                kind=ConsoleDispatchRecoveryKind.QUARANTINED,
                assistant_message_id="",
                conversation_id=conversation_id,
                visible_copy=(
                    "Dispatch recovery is unavailable because persisted ownership "
                    "is invalid."
                ),
                actions=(),
                error_code="checkpoint_read_error",
            )
        if recovery is None or not recovery.recovery_needed:
            self._dispatch_recoveries_by_session.pop(session_id, None)
            self._dispatch_recovery_message_baselines.pop(session_id, None)
            self._dispatch_recovery_queue_hydration_pending.discard(session_id)
        else:
            self._dispatch_recoveries_by_session[session_id] = recovery
            checkpoint = recovery.checkpoint
            if (
                checkpoint is not None
                and checkpoint.origin == "queued"
                and checkpoint.queue_entry_id is not None
            ):
                self._dispatch_recovery_queue_hydration_pending.add(session_id)
            else:
                self._dispatch_recovery_queue_hydration_pending.discard(session_id)
        return recovery

    def dispatch_recovery_for_session(
        self, session_id: str | None
    ) -> ConsoleDispatchRecoveryState | None:
        """Return one immutable app-runtime recovery owner, if present."""

        if not isinstance(session_id, str) or not session_id:
            return None
        with self._preparation_lock:
            return self._dispatch_recoveries_by_session.get(session_id)

    def dispatch_recovery_for_presentation(
        self, session_id: str | None
    ) -> ConsoleDispatchRecoveryState | None:
        """Return only an owner that currently needs user recovery."""

        recovery = self.dispatch_recovery_for_session(session_id)
        if recovery is None or not recovery.recovery_needed:
            return None
        return recovery

    def dispatch_recovery_blocks_submission(self, session_id: str | None) -> bool:
        """Return whether one UNRESOLVED source-local owner blocks the next send.

        TASK-22000 (owner decision, 2026-08-24): this reads the *presentation*
        owner, not the raw one. Only an owner the user is actually being shown
        a recovery card for can refuse a send -- a healthy in-flight run
        (``runtime_active=True, recovery_needed=False``) never does.

        The original TASK-19900.3 predicate keyed on ``kind`` alone, so the
        app's own live durable turn refused submission for its whole duration.
        ADR-046 / TASK-14808 say the opposite: an accepted live turn re-labels
        Send to "Queue" and admits the draft as a FIFO follow-up. The user got
        a button labelled "Queue" that was greyed out. Reading the presentation
        owner makes the two agree by construction: nothing invisible refuses.

        What the gate was actually guarding is untouched, because every one of
        those states carries ``recovery_needed=True``:

        * a checkpoint restored from a previous app run
          (``_hydrate_dispatch_recovery`` stores an owner only when it needs
          recovery), which would otherwise hit the repository's
          "active dispatch checkpoint" refusal on the next send;
        * a live owner whose terminal settlement failed
          (``mark_dispatch_recovery_needed`` /
          ``_restore_dispatch_recovery_after_settlement_failure``) -- note its
          run state is ``BLOCKED``, which ``is_send_allowed`` *permits*, so
          this gate is the only thing standing between the user and a raw
          ``RuntimeError`` from a second durable owner;
        * ``QUARANTINED`` ownership that could not be read at all.

        A healthy in-flight owner needs no gate here: its run state is
        VALIDATING/STREAMING, so ``_active_run_rejection`` already refuses a
        second manual turn, and a queued follow-up is only *submitted* from
        ``_drain_waiting``, which runs after the previous turn reaches a
        terminal status -- by which point settlement has popped this owner.

        Args:
            session_id: Native Console session id. ``None`` -- and equally an
                empty string or any non-``str`` -- means "no session to have
                an owner", which ``dispatch_recovery_for_session`` answers
                with ``None``, so the gate is open. Callers on a screen with
                no active session therefore need no guard of their own.

        Returns:
            ``True`` only when that session has a recovery owner the user is
            currently being shown a card for AND its kind is one of the five
            unresolved source-local kinds above. ``False`` for no owner, for
            a healthy in-flight owner (``recovery_needed=False``), and for
            the three kinds outside that set (``REMOTE_ACCEPTED``,
            ``REMOTE_DISPATCH_STARTED``, ``CONTINUATION``) -- i.e. the send
            is admitted.
        """

        recovery = self.dispatch_recovery_for_presentation(session_id)
        return recovery is not None and recovery.kind in {
            ConsoleDispatchRecoveryKind.ACCEPTED,
            ConsoleDispatchRecoveryKind.DISPATCH_STARTED,
            ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
            ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
            ConsoleDispatchRecoveryKind.QUARANTINED,
        }

    def dispatch_recovery_needs_queue_hydration(self, session_id: str) -> bool:
        """Return whether restore published a queued owner not yet projected."""

        with self._preparation_lock:
            return session_id in self._dispatch_recovery_queue_hydration_pending

    def mark_dispatch_recovery_queue_hydrated(self, session_id: str) -> None:
        """Acknowledge projection of one restored queued recovery owner."""

        with self._preparation_lock:
            self._dispatch_recovery_queue_hydration_pending.discard(session_id)

    def publish_durable_dispatch_checkpoint(
        self,
        session_id: str,
        checkpoint: ConsoleDispatchCheckpoint,
        *,
        in_flight: bool,
    ) -> ConsoleDispatchRecoveryState:
        """Publish the exact committed durable owner into app-runtime state."""

        self._session_or_raise(session_id)
        recovery = console_dispatch_recovery_from_checkpoint(
            checkpoint,
            in_flight=in_flight,
        )
        recovery = recovery.with_runtime_truth(
            runtime_active=True,
            recovery_needed=False,
        )
        with self._preparation_lock:
            current = self._dispatch_recoveries_by_session.get(session_id)
            if current is not None and (
                current.assistant_message_id != recovery.assistant_message_id
                or current.conversation_id != recovery.conversation_id
            ):
                raise RuntimeError("Durable dispatch recovery owner changed.")
            if in_flight:
                message = self._message_or_raise(recovery.assistant_message_id)
                self._dispatch_recovery_message_baselines[session_id] = self._snapshot(
                    message
                )
            self._dispatch_recoveries_by_session[session_id] = recovery
        return recovery

    def register_ephemeral_dispatch_recovery(
        self,
        session_id: str,
        *,
        user_message_id: str,
        assistant_message_id: str,
        preparation_id: str,
        attempt_id: str,
        checkpoint_state: ConsoleDispatchCheckpointState,
        origin: str,
        queue_entry_id: str | None,
        frozen_authority: ConsoleTurnLibraryAuthority,
        resolved_destination: ConsoleResolvedDestination,
        reconstructability: ConsoleDispatchReconstructability,
        runtime_active: bool = False,
    ) -> ConsoleDispatchRecoveryState:
        """Install the no-SQL analogue of one accepted dispatch checkpoint."""

        session = self._session_or_raise(session_id)
        if type(runtime_active) is not bool:
            raise TypeError("runtime_active must be a bool")
        if not session.ephemeral:
            raise RuntimeError("Only a temporary session can own ephemeral recovery.")
        user = self._message_or_raise(user_message_id)
        assistant = self._message_or_raise(assistant_message_id)
        if (
            self._message_session_index.get(user.id) != session_id
            or self._message_session_index.get(assistant.id) != session_id
            or user.role is not ConsoleMessageRole.USER
            or assistant.role is not ConsoleMessageRole.ASSISTANT
        ):
            raise RuntimeError("Ephemeral dispatch owners changed.")
        checkpoint = ConsoleDispatchCheckpoint(
            assistant_message_id=assistant_message_id,
            user_message_id=user_message_id,
            conversation_id=session_id,
            preparation_id=preparation_id,
            attempt_id=attempt_id,
            state=checkpoint_state,
            checkpoint_revision=1,
            user_message_version=1,
            assistant_message_version=1,
            origin=origin,
            queue_entry_id=queue_entry_id,
            frozen_authority=frozen_authority,
            resolved_destination=resolved_destination,
            reconstructability=reconstructability,
        )
        recovery = console_dispatch_recovery_from_checkpoint(
            checkpoint,
            ephemeral=True,
        ).with_runtime_truth(
            runtime_active=runtime_active,
            recovery_needed=not runtime_active,
        )
        with self._preparation_lock:
            current = self._dispatch_recoveries_by_session.get(session_id)
            if current is not None:
                if current == recovery:
                    return current
                raise RuntimeError("Ephemeral dispatch recovery is already owned.")
            self._dispatch_recoveries_by_session[session_id] = recovery
        return recovery

    def claim_dispatch_recovery_action(
        self,
        session_id: str,
        action_id: ConsoleDispatchRecoveryActionId,
    ) -> ConsoleDispatchRecoveryState | None:
        """Disable repeated intents and return only one exact action claimant."""

        if not isinstance(action_id, ConsoleDispatchRecoveryActionId):
            raise TypeError("action_id must be a ConsoleDispatchRecoveryActionId")
        with self._preparation_lock:
            current = self._dispatch_recoveries_by_session.get(session_id)
            if current is None or current.in_flight or not current.recovery_needed:
                return None
            action = next(
                (item for item in current.actions if item.action_id is action_id),
                None,
            )
            if action is None or not action.enabled:
                return None
            try:
                message = self._message_or_raise(current.assistant_message_id)
            except KeyError:
                return None
            self._dispatch_recovery_message_baselines[session_id] = self._snapshot(
                message
            )
            updated = current.with_in_flight(True)
            self._dispatch_recoveries_by_session[session_id] = updated
            return updated

    def release_dispatch_recovery_action(
        self,
        session_id: str,
        assistant_message_id: str,
    ) -> bool:
        """Re-enable one failed in-flight intent without changing ownership."""

        with self._preparation_lock:
            current = self._dispatch_recoveries_by_session.get(session_id)
            if current is None or current.assistant_message_id != assistant_message_id:
                return False
            if current.in_flight:
                self._dispatch_recoveries_by_session[session_id] = (
                    current.with_in_flight(False)
                )
            baseline = self._dispatch_recovery_message_baselines.pop(session_id, None)
        if baseline is not None:
            try:
                message = self._message_or_raise(assistant_message_id)
            except KeyError:
                return current.in_flight
            for message_field in fields(message):
                setattr(
                    message,
                    message_field.name,
                    getattr(baseline, message_field.name),
                )
            self._stream_chunks_by_message.pop(message.id, None)
            self._stream_materialized_counts.pop(message.id, None)
            self._bump_payload_revision(session_id)
        return current.in_flight

    def mark_dispatch_recovery_needed(
        self,
        session_id: str,
        assistant_message_id: str,
    ) -> bool:
        """Restore one exact owner and expose it as unresolved recovery."""

        released = self.release_dispatch_recovery_action(
            session_id,
            assistant_message_id,
        )
        with self._preparation_lock:
            current = self._dispatch_recoveries_by_session.get(session_id)
            if current is None or current.assistant_message_id != assistant_message_id:
                return False
            self._dispatch_recoveries_by_session[session_id] = (
                current.with_runtime_truth(
                    runtime_active=False,
                    recovery_needed=True,
                )
            )
            checkpoint = current.checkpoint
            if (
                checkpoint is not None
                and checkpoint.origin == "queued"
                and checkpoint.queue_entry_id is not None
            ):
                self._dispatch_recovery_queue_hydration_pending.add(session_id)
        return released or current.recovery_needed is False

    def transition_dispatch_recovery_for_retry(
        self,
        session_id: str,
        *,
        assistant_message_id: str,
        new_attempt_id: str,
    ) -> ConsoleDispatchRecoveryState | None:
        """CAS a claimed owner immediately before its explicit provider retry."""

        with self._preparation_lock:
            current = self._dispatch_recoveries_by_session.get(session_id)
            if (
                current is None
                or not current.in_flight
                or current.assistant_message_id != assistant_message_id
                or current.checkpoint is None
            ):
                return None
            checkpoint = current.checkpoint
            if current.kind in {
                ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
                ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
            }:
                updated_checkpoint = replace(
                    checkpoint,
                    state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
                    checkpoint_revision=checkpoint.checkpoint_revision + 1,
                    assistant_message_version=checkpoint.assistant_message_version + 1,
                    attempt_id=new_attempt_id,
                )
                updated = console_dispatch_recovery_from_checkpoint(
                    updated_checkpoint,
                    ephemeral=True,
                    in_flight=True,
                ).with_runtime_truth(
                    runtime_active=True,
                    recovery_needed=current.recovery_needed,
                )
                self._dispatch_recoveries_by_session[session_id] = updated
                return updated
        repository = getattr(
            self.persistence,
            "console_dispatch_repository",
            None,
        )
        if repository is None:
            return None
        result = repository.cas_state(
            ConsoleDispatchTransition(
                assistant_message_id=checkpoint.assistant_message_id,
                expected_state=checkpoint.state,
                expected_checkpoint_revision=checkpoint.checkpoint_revision,
                expected_user_message_version=checkpoint.user_message_version,
                expected_assistant_message_version=checkpoint.assistant_message_version,
                new_state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
                new_attempt_id=new_attempt_id,
            )
        )
        if (
            result.status is not ConsoleDispatchResultStatus.COMMITTED
            or result.checkpoint is None
        ):
            return None
        updated = console_dispatch_recovery_from_checkpoint(
            result.checkpoint,
            in_flight=True,
        ).with_runtime_truth(
            runtime_active=True,
            recovery_needed=current.recovery_needed,
        )
        with self._preparation_lock:
            if self._dispatch_recoveries_by_session.get(session_id) is not current:
                raise RuntimeError("Dispatch recovery owner changed after CAS.")
            self._dispatch_recoveries_by_session[session_id] = updated
        return updated

    def begin_ephemeral_dispatch(
        self,
        session_id: str,
        *,
        assistant_message_id: str,
        new_attempt_id: str,
    ) -> ConsoleDispatchRecoveryState | None:
        """Mark a newly accepted ephemeral turn started without UI action gating."""

        with self._preparation_lock:
            current = self._dispatch_recoveries_by_session.get(session_id)
            if (
                current is None
                or current.in_flight
                or current.kind is not ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED
                or current.assistant_message_id != assistant_message_id
            ):
                return None
            try:
                message = self._message_or_raise(assistant_message_id)
            except KeyError:
                return None
            self._dispatch_recovery_message_baselines[session_id] = self._snapshot(
                message
            )
            self._dispatch_recoveries_by_session[session_id] = current.with_in_flight(
                True
            )
        transitioned = self.transition_dispatch_recovery_for_retry(
            session_id,
            assistant_message_id=assistant_message_id,
            new_attempt_id=new_attempt_id,
        )
        if transitioned is None:
            self.release_dispatch_recovery_action(session_id, assistant_message_id)
        return transitioned

    def prepare_dispatch_recovery_message(
        self,
        session_id: str,
        assistant_message_id: str,
    ) -> ConsoleChatMessage:
        """Reset only the exact existing assistant owner for retry streaming."""

        recovery = self.dispatch_recovery_for_session(session_id)
        message = self._message_or_raise(assistant_message_id)
        if (
            recovery is None
            or not recovery.in_flight
            or recovery.assistant_message_id != assistant_message_id
            or self._message_session_index.get(assistant_message_id) != session_id
            or message.role is not ConsoleMessageRole.ASSISTANT
        ):
            raise RuntimeError("Dispatch retry assistant owner changed.")
        message.content = ""
        message.status = "pending"
        message.thinking = None
        message.opaque_thinking_json = None
        message.thinking_warning = None
        message.thinking_actions_enabled = True
        message.usage = None
        message.provider_continuation = None
        message.provider_continuation_warning = None
        message.provider_continuation_remote = False
        message.provider_continuation_message_version = None
        message.provider_continuation_actions_enabled = True
        message.assistant_generation_state = None
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        self._bump_payload_revision(session_id)
        return self._snapshot(message)

    def settle_dispatch_recovery(
        self,
        session_id: str,
        *,
        assistant_message_id: str,
        terminal_state: str,
        content: str,
        metadata_json: str | None = None,
        provider_continuation_json: str | None = None,
        provider_continuation: ProviderContinuationCheckpoint | None = None,
    ) -> bool:
        """Settle one claimed durable or ephemeral owner without a second write."""

        with self._preparation_lock:
            current = self._dispatch_recoveries_by_session.get(session_id)
            if (
                current is None
                or not current.in_flight
                or current.assistant_message_id != assistant_message_id
                or current.checkpoint is None
            ):
                return False
            checkpoint = current.checkpoint
            ephemeral = current.kind in {
                ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
                ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
            }
        try:
            message = self._message_or_raise(assistant_message_id)
        except KeyError:
            message = None
        committed_message_version: int | None = None
        if not ephemeral:
            repository = getattr(
                self.persistence,
                "console_dispatch_repository",
                None,
            )
            if repository is None:
                return False
            try:
                result = repository.settle_with_assistant(
                    ConsoleAssistantSettlement(
                        assistant_message_id=assistant_message_id,
                        expected_checkpoint_state=checkpoint.state,
                        expected_checkpoint_revision=checkpoint.checkpoint_revision,
                        expected_user_message_version=checkpoint.user_message_version,
                        expected_assistant_message_version=(
                            checkpoint.assistant_message_version
                        ),
                        terminal_state=terminal_state,
                        content=content,
                        metadata_json=metadata_json,
                        usage_json=(
                            message.usage.to_json()
                            if message is not None and message.usage is not None
                            else None
                        ),
                        provider_continuation_json=provider_continuation_json,
                        thinking_blocks_json=(
                            dump_thinking_blocks_json(message.thinking)
                            if message is not None
                            else None
                        ),
                    )
                )
            except Exception:
                return False
            if result.status is not ConsoleDispatchResultStatus.COMMITTED:
                return False
            committed_message_version = result.committed_message_version
        if message is not None:
            message.content = content
            message.status = terminal_state
            message.assistant_generation_state = terminal_state
            message.provider_continuation = provider_continuation
            message.provider_continuation_message_version = committed_message_version
            message.provider_continuation_actions_enabled = False
            self._stream_chunks_by_message.pop(message.id, None)
            self._stream_materialized_counts.pop(message.id, None)
            self._bump_payload_revision(session_id)
        with self._preparation_lock:
            if self._dispatch_recoveries_by_session.get(session_id) is not current:
                raise RuntimeError("Dispatch recovery owner changed during settlement.")
            self._dispatch_recoveries_by_session.pop(session_id, None)
            self._dispatch_recovery_message_baselines.pop(session_id, None)
            self._dispatch_recovery_queue_hydration_pending.discard(session_id)
        return True

    def _settle_owned_dispatch_terminal(
        self,
        message: ConsoleChatMessage,
        terminal_state: str,
    ) -> bool:
        """Use the repository settlement instead of a separate content update."""

        session_id = self._message_session_index[message.id]
        recovery = self.dispatch_recovery_for_session(session_id)
        if recovery is None or recovery.assistant_message_id != message.id:
            return False
        if not recovery.in_flight:
            raise ConsoleDispatchSettlementError(
                "Dispatch terminal settlement previously failed."
            )
        metadata_json = None
        if message.video_metadata is not None:
            metadata_json = message.video_metadata.to_json()
        elif message.metadata is not None and not message.metadata.is_empty:
            metadata_json = message.metadata.to_json()
        if not self.settle_dispatch_recovery(
            session_id,
            assistant_message_id=message.id,
            terminal_state=terminal_state,
            content=message.content,
            metadata_json=metadata_json,
        ):
            raise ConsoleDispatchSettlementError(
                "Durable dispatch terminal settlement failed."
            )
        return True

    def _hydrate_provider_continuations_from_persistence(
        self,
        session_id: str,
        conversation_id: str,
        nodes: list[ConsoleChatMessage],
        *,
        remote_active: bool = False,
    ) -> list[ConsoleChatMessage]:
        """Tolerantly attach private checkpoints without exposing their data."""
        database = getattr(self.persistence, "db", None) if self.persistence else None
        getter = getattr(database, "get_messages_for_conversation", None)
        if not callable(getter):
            self._quarantine_continuation_hydration(session_id, conversation_id)
            return nodes
        try:
            rows = getter(conversation_id, limit=100_000)
        except Exception:
            logger.warning("Console continuation restore was unavailable.")
            self._quarantine_continuation_hydration(session_id, conversation_id)
            return nodes
        by_persisted_id = {
            node.persisted_message_id: node
            for node in nodes
            if node.persisted_message_id is not None
        }
        for row in rows:
            persisted_id = str(row.get("id") or "")
            safe = read_provider_continuation_json(
                row.get("provider_continuation_json")
            )
            thinking = read_thinking_blocks_json(row.get("thinking_blocks_json"))
            node = by_persisted_id.get(persisted_id)
            if node is None and (
                safe.checkpoint is not None
                or safe.warning
                or thinking.envelope is not None
                or thinking.warning
            ):
                node = ConsoleChatMessage(
                    id=persisted_id,
                    role=ConsoleMessageRole.ASSISTANT,
                    content=str(row.get("content") or ""),
                    persisted_message_id=persisted_id,
                    parent_message_id=(
                        str(row["parent_message_id"])
                        if row.get("parent_message_id") is not None
                        else None
                    ),
                )
                nodes.append(node)
                by_persisted_id[persisted_id] = node
            if node is None:
                continue
            node.parent_message_id = (
                str(row["parent_message_id"])
                if row.get("parent_message_id") is not None
                else None
            )
            node.provider_continuation = safe.checkpoint
            node.provider_continuation_warning = safe.warning
            node.provider_continuation_remote = bool(
                safe.checkpoint is not None and remote_active
            )
            version = row.get("version")
            node.provider_continuation_message_version = (
                version if type(version) is int else None
            )
            node.assistant_generation_state = row.get("assistant_generation_state")
            node.provider_continuation_actions_enabled = False
            node.thinking = thinking.envelope
            node.opaque_thinking_json = thinking.opaque_json
            node.thinking_warning = thinking.warning
            node.thinking_actions_enabled = (
                thinking.generation_actions_enabled and thinking.warning is None
            )
        return nodes

    def _quarantine_continuation_hydration(
        self,
        session_id: str,
        conversation_id: str,
    ) -> None:
        """Keep an unreadable continuation owner blocking until a fresh restore."""

        with self._preparation_lock:
            current = self._dispatch_recoveries_by_session.get(session_id)
            if (
                current is None
                or current.kind is not ConsoleDispatchRecoveryKind.CONTINUATION
            ):
                return
            self._dispatch_recoveries_by_session[session_id] = (
                ConsoleDispatchRecoveryState(
                    kind=ConsoleDispatchRecoveryKind.QUARANTINED,
                    assistant_message_id=current.assistant_message_id,
                    conversation_id=conversation_id,
                    visible_copy=(
                        "Continuation recovery is unavailable; reload the conversation."
                    ),
                    actions=(),
                    error_code="continuation_hydration_error",
                )
            )
            self._dispatch_recovery_message_baselines.pop(session_id, None)
            self._dispatch_recovery_queue_hydration_pending.discard(session_id)

    def _normalize_restored_provider_continuation(
        self, session_id: str, conversation_id: str
    ) -> None:
        """Normalize and rebind one restored active ADR-063 owner before actions."""
        recovery = self.dispatch_recovery_for_session(session_id)
        if (
            recovery is None
            or recovery.kind is not ConsoleDispatchRecoveryKind.CONTINUATION
        ):
            return
        message = self._nodes_by_session.get(session_id, {}).get(
            recovery.assistant_message_id
        )
        repository = getattr(self.persistence, "console_dispatch_repository", None)
        snapshot_reader = getattr(
            repository, "provider_continuation_owner_snapshot", None
        )
        normalizer = getattr(repository, "normalize_provider_continuation_owner", None)
        if (
            message is None
            or message.provider_continuation is None
            or message.provider_continuation.state != "active"
            or not callable(snapshot_reader)
            or not callable(normalizer)
        ):
            if message is not None:
                message.provider_continuation_actions_enabled = False
                message.provider_continuation_message_version = None
            with self._preparation_lock:
                current = self._dispatch_recoveries_by_session.get(session_id)
                if current is recovery:
                    self._dispatch_recoveries_by_session.pop(session_id, None)
                    self._dispatch_recovery_message_baselines.pop(session_id, None)
                    self._dispatch_recovery_queue_hydration_pending.discard(session_id)
            return
        original = message.provider_continuation
        original_json = dump_provider_continuation_json(original)
        try:
            for _attempt in range(2):
                observed = snapshot_reader(
                    conversation_id=conversation_id,
                    assistant_message_id=message.persisted_message_id or message.id,
                )
                if (
                    not isinstance(observed, Mapping)
                    or observed.get("checkpoint") != original
                    or observed.get("canonical") != original_json
                    or type(observed.get("version")) is not int
                ):
                    message.provider_continuation_actions_enabled = False
                    message.provider_continuation_message_version = None
                    message.provider_continuation_warning = (
                        "Continuation changed during restore; reload before recovery."
                    )
                    return
                observed_version = int(observed["version"])
                observed_state = observed.get("state")
                if observed_state == "continuation_active":
                    message.assistant_generation_state = "continuation_active"
                    message.provider_continuation_message_version = observed_version
                    message.provider_continuation_actions_enabled = True
                    return
                try:
                    result = normalizer(
                        conversation_id=conversation_id,
                        assistant_message_id=(
                            message.persisted_message_id or message.id
                        ),
                        expected_message_version=observed_version,
                        expected_state=observed_state,
                        provider_continuation_json=original_json,
                    )
                except Exception:
                    fresh = snapshot_reader(
                        conversation_id=conversation_id,
                        assistant_message_id=(
                            message.persisted_message_id or message.id
                        ),
                    )
                    if (
                        isinstance(fresh, Mapping)
                        and fresh.get("checkpoint") == original
                        and fresh.get("canonical") == original_json
                        and fresh.get("version") == observed_version
                        and fresh.get("state") == observed_state
                    ):
                        message.provider_continuation_message_version = observed_version
                        message.provider_continuation_actions_enabled = True
                        message.provider_continuation_warning = (
                            "Continuation normalization was rolled back; the exact "
                            "durable owner was confirmed."
                        )
                    return
                if (
                    result.status is ConsoleDispatchResultStatus.COMMITTED
                    and type(result.committed_message_version) is int
                ):
                    message.assistant_generation_state = "continuation_active"
                    message.provider_continuation_message_version = (
                        result.committed_message_version
                    )
                    message.provider_continuation_actions_enabled = True
                    message.provider_continuation_warning = None
                    return
        finally:
            with self._preparation_lock:
                current = self._dispatch_recoveries_by_session.get(session_id)
                if current is recovery:
                    self._dispatch_recoveries_by_session.pop(session_id, None)
                    self._dispatch_recovery_message_baselines.pop(session_id, None)
                    self._dispatch_recovery_queue_hydration_pending.discard(session_id)

    def _reconcile_restored_chat_sync_intents(
        self, session_id: str, conversation_id: str
    ) -> None:
        """Project exact current unbridged Chat intents during normal restore."""
        if self.sync_v2_server_profile_id is None or self.sync_v2_chat_producer is None:
            return
        database = getattr(self.persistence, "db", None) if self.persistence else None
        enumerate_intents = getattr(
            database, "list_current_committed_chat_sync_intents", None
        )
        if not callable(enumerate_intents):
            return
        try:
            intents = enumerate_intents(conversation_id)
        except Exception:
            logger.warning("Console continuation sync reconciliation was unavailable.")
            return
        producer = self.sync_v2_chat_producer
        for intent in intents:
            if not isinstance(intent, Mapping):
                continue
            operation = intent.get("operation")
            message_id = intent.get("message_id")
            message_version = intent.get("message_version")
            payload_hash = intent.get("payload_hash")
            if (
                operation not in {"upsert", "delete"}
                or type(message_id) is not str
                or type(message_version) is not int
                or type(payload_hash) is not str
            ):
                continue
            reconcile = getattr(
                producer,
                "reconcile_chat_message_delete_intent"
                if operation == "delete"
                else "reconcile_chat_message_intent",
                None,
            )
            if not callable(reconcile):
                continue
            try:
                result = reconcile(
                    server_profile_id=self.sync_v2_server_profile_id,
                    authenticated_principal_id=(
                        self.sync_v2_authenticated_principal_id
                    ),
                    workspace_scope=self.sync_v2_workspace_scope,
                    message_id=message_id,
                    message_version=message_version,
                    payload_hash=payload_hash,
                )
            except Exception:
                logger.warning("Failed to reconcile restored Chat sync intent")
                continue
            if not isinstance(result, Mapping) or result.get("status") != "enqueued":
                for message in self._nodes_by_session.get(session_id, {}).values():
                    if message.persisted_message_id == message_id:
                        message.provider_continuation_warning = (
                            "Portable continuation reconciliation is pending."
                        )
                        break

    def _restore_speech_preferences(self, session: ConsoleChatSession) -> None:
        """Fail closed while hydrating conversation-owned speech preferences."""
        if self.persistence is None or session.persisted_conversation_id is None:
            return
        reader = getattr(
            self.persistence,
            "get_conversation_speech_preferences",
            None,
        )
        if not callable(reader):
            return
        try:
            restored = reader(session.persisted_conversation_id)
        except Exception as exc:
            logger.warning(
                "Failed to restore Console reply-speech preferences "
                "(exception_type={}).",
                type(exc).__name__,
            )
            return
        if isinstance(restored, ConsoleSpeechPreferences):
            session.speech_preferences = restored

    def _hydrate_generation_metadata_from_persistence(self, session_id: str) -> None:
        """Batch-fetch and apply generation-metadata sidecar rows on resume.

        The store-level counterpart of Task 9's manual round-trip: every
        caller of ``restore_persisted_session`` (a saved-conversation resume
        from the workspace rail, and -- after an app restart, since that is
        the only way back into a persisted conversation -- effectively every
        production reload) gets this for free instead of each caller having
        to remember to drive ``get_generation_metadata_for_messages`` +
        ``hydrate_generation_metadata`` itself. Optional and probe-guarded
        like the sibling attachment batch-fetch: a ``persistence`` with no
        ``get_generation_metadata_for_messages`` (older fakes, or a future
        persistence shape without one) leaves resumed messages
        metadata-only, matching this store's other graceful-degradation
        seams. Covers every restored node (all branches, on- and off-path)
        in a SINGLE batched call, not one round trip per message.
        """
        if self.persistence is None:
            return
        getter = getattr(self.persistence, "get_generation_metadata_for_messages", None)
        if not callable(getter):
            return
        nodes = self._nodes_by_session.get(session_id, {})
        message_ids = [
            message.persisted_message_id
            for message in nodes.values()
            if message.persisted_message_id is not None
        ]
        if not message_ids:
            return
        try:
            rows_by_message = getter(message_ids)
        except Exception:
            logger.opt(exception=True).warning(
                "Console resume generation-metadata batch fetch failed."
            )
            return
        if not isinstance(rows_by_message, dict):
            return
        self.hydrate_generation_metadata(session_id, rows_by_message)

    def apply_resume_marker_overlay(
        self, session_id: str, messages: Sequence[ConsoleChatMessage]
    ) -> None:
        """Overlay resume-derived, display-only TOOL markers onto the view.

        The active-path VIEW (``_messages_by_session``) is normally the
        single-writer output of ``_recompute_active_path`` (tree nodes only).
        On resume, agent TOOL markers are re-derived from ``AgentRunsDB`` (they
        are ``persist=False`` and never tree nodes) and interleaved into the
        rendered transcript for display; this installs that interleaved list as
        the current view. Real (non-marker) rows are re-resolved to their LIVE
        tree nodes so a later render still observes them; TOOL markers are the
        given snapshots and are registered in the session index so
        ``close_session`` sweeps them.

        The overlay is transient by design: the next ``_recompute_active_path``
        (any tree mutation -- send, swipe, delete) rebuilds the view from live
        tree nodes and drops the markers, exactly as live markers are ephemeral
        in Phase A. When ``messages`` carries no markers (no agent runs) this is
        equivalent to the freshly recomputed view.
        """
        self._session_or_raise(session_id)
        nodes = self._nodes_by_session.get(session_id, {})
        overlay: list[ConsoleChatMessage] = []
        # TASK-1842: re-register resumed markers against the node they follow.
        # Resume re-derives them from AgentRunsDB (`inject_resume_agent_markers`)
        # and lands here, which writes the view DIRECTLY -- bypassing
        # `append_message`. Without this, a resumed session's trace survived
        # exactly until the first `_recompute_active_path`, which is the same
        # data loss by a different door. Reset first so a re-resume cannot
        # stack a second copy of the same DB-derived markers.
        self._tool_markers_by_session[session_id] = []
        last_node_id: str | None = None
        for message in messages:
            if message.role is ConsoleMessageRole.TOOL:
                self._message_session_index.setdefault(message.id, session_id)
                self._tool_markers_by_session[session_id].append(
                    (last_node_id, message)
                )
                overlay.append(message)
            else:
                resolved = nodes.get(message.id, message)
                last_node_id = resolved.id
                overlay.append(resolved)
        self._messages_by_session[session_id] = overlay

    def switch_session(self, session_id: str) -> ConsoleChatSession:
        """Activate an existing session."""
        session = self._session_or_raise(session_id)
        self._activate_session(session.id)
        return session

    def rename_session(
        self, session_id: str, title: str
    ) -> tuple[ConsoleChatSession, bool]:
        """Rename a native Console session, persisting a saved conversation's title.

        TASK-341: the tab IS the conversation for a resumed saved
        conversation — renaming only the in-memory session looked successful
        (tab + transcript header updated) but evaporated on restart.

        Args:
            session_id: Native Console session ID to rename.
            title: New title; surrounding whitespace is trimmed.

        Returns:
            ``(session, persisted)`` — the in-memory rename is always
            applied. ``persisted`` is ``False`` when the session has a saved
            conversation whose durable title update did not happen: the
            persistence call raised, returned falsy (e.g. an optimistic-lock
            version check refused the write), or the persistence object has
            no ``update_conversation_title`` seam at all.

        Raises:
            ValueError: If the trimmed title is blank.
            KeyError: If no session with ``session_id`` exists.
        """
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("Console chat session title cannot be blank.")
        session = self._session_or_raise(session_id)
        if session.title != normalized_title:
            session.has_user_work = True
            session.canonical_settings_baseline = None
        session.title = normalized_title
        persisted = True
        if (
            session.persisted_conversation_id is not None
            and self.persistence is not None
        ):
            update_title = getattr(self.persistence, "update_conversation_title", None)
            if not callable(update_title):
                # A saved conversation with no durable rename seam: claiming
                # persisted=True here would recreate the original silent-loss
                # bug for exactly the sessions this fix targets.
                persisted = False
            else:
                try:
                    persisted = bool(
                        update_title(
                            conversation_id=session.persisted_conversation_id,
                            title=normalized_title,
                        )
                    )
                except Exception:
                    persisted = False
                    logger.bind(
                        session_id=session_id,
                        conversation_id=session.persisted_conversation_id,
                    ).exception(
                        "Failed to persist Console session title; "
                        "in-memory session keeps the applied value."
                    )
        return session, persisted

    def close_session(self, session_id: str) -> ConsoleChatSession | None:
        """Close a native Console session and activate a neighboring session.

        Args:
            session_id: Native Console session ID to close.

        Returns:
            The session activated after closing, or ``None`` when no sessions remain.
        """
        self._session_or_raise(session_id)
        recovery = self.dispatch_recovery_for_session(session_id)
        if (
            recovery is not None
            and recovery.recovery_needed
            and recovery.kind
            in {
                ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
                ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
            }
        ):
            raise RuntimeError(
                "Finish or discard the pending turn before closing this chat."
            )
        preparation = self.preparation_for_session(session_id)
        if preparation is not None and preparation.state in {
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.READY,
            ConsoleTurnPreparationState.PAUSED,
        }:
            self.cancel_preparation(
                session_id,
                preparation.preparation_id,
                expected_state=preparation.state,
            )
        session_ids = list(self._sessions.keys())
        closed_index = session_ids.index(session_id)

        self._purge_session_runtime_state(session_id)

        if self.active_session_id != session_id:
            return self._sessions.get(self.active_session_id or "")

        remaining_sessions = list(self._sessions.values())
        if not remaining_sessions:
            self._activate_session(None)
            return None

        next_index = min(closed_index, len(remaining_sessions) - 1)
        next_session = remaining_sessions[next_index]
        self._activate_session(next_session.id)
        return next_session

    def _purge_session_runtime_state(self, session_id: str) -> None:
        """Delete one session's exact process-local ownership without DB writes."""

        # Purge EVERY message the session owns, not just the active-path view:
        # off-path tree nodes and dropped display-only TOOL markers both live in
        # ``_message_session_index`` (a superset of ``_nodes_by_session`` for the
        # session), so it is the authoritative set of owned ids to sweep.
        owned_message_ids = [
            message_id
            for message_id, owner in list(self._message_session_index.items())
            if owner == session_id
        ]
        for message_id in owned_message_ids:
            self.clear_terminal_citation_state(message_id)
            self._message_session_index.pop(message_id, None)
            self._stream_chunks_by_message.pop(message_id, None)
            self._stream_materialized_counts.pop(message_id, None)
            self._pending_persistence_message_ids.discard(message_id)
            self._variant_stream_bases.pop(message_id, None)
            self._variant_restored_message_ids.discard(message_id)
            self._failed_retry_message_ids.discard(message_id)
            self._message_speech_revisions.pop(message_id, None)
            self._message_completion_generations.pop(message_id, None)
            self._native_parent_by_message.pop(message_id, None)
            self._roleplay_message_projection_candidates.pop(message_id, None)
            self._exchange_blob_cache.pop(message_id, None)
            self._abandoned_exchange_run_tags.pop(message_id, None)
            self._character_emote_captures.pop(message_id, None)
            self._trajectory_timing.pop(message_id, None)
            self._trajectory_written_ids.discard(message_id)
            self._pending_trajectory_tool_rows.pop(message_id, None)
            self._pending_trajectory_event_rows.pop(message_id, None)

        unanchored = self._pending_trajectory_tool_rows.get("__unanchored__")
        if unanchored is not None:
            retained = [
                entry
                for entry in unanchored
                if entry.get("session_id") != session_id
            ]
            if retained:
                self._pending_trajectory_tool_rows["__unanchored__"] = retained
            else:
                self._pending_trajectory_tool_rows.pop("__unanchored__", None)

        self._messages_by_session.pop(session_id, None)
        self._tool_markers_by_session.pop(session_id, None)
        self._nodes_by_session.pop(session_id, None)
        self._children_by_parent.pop(session_id, None)
        self._active_leaf_by_session.pop(session_id, None)
        self._context_summary_by_session.pop(session_id, None)
        self._deferred_project_instruction_state_session_ids.discard(session_id)
        self._roleplay_system_projection_candidates.pop(session_id, None)
        self._payload_revisions.pop(session_id, None)
        self._conversation_context_epochs.pop(session_id, None)
        self._speech_preference_epochs.pop(session_id, None)
        self._character_emote_feed_by_session.pop(session_id, None)
        self._unresolved_promotion_operations.pop(session_id, None)
        self._dispatch_recoveries_by_session.pop(session_id, None)
        self._dispatch_recovery_message_baselines.pop(session_id, None)
        self._dispatch_recovery_queue_hydration_pending.discard(session_id)
        self._pending_workspace_projections.pop(session_id, None)
        self._session_turn_ids.pop(session_id, None)
        if self.library_policy_coordinator is not None:
            self.library_policy_coordinator.unregister_holder(session_id)
        self._sessions.pop(session_id, None)
        with self._preparation_lock:
            preparation = self._preparations_by_session.get(session_id)
            if preparation is not None:
                fingerprint = self._durable_fingerprint_by_preparation.get(
                    preparation.preparation_id
                )
                if fingerprint is not None:
                    self.retire_durable_acceptance(
                        preparation.preparation_id, fingerprint
                    )
                else:
                    self.discard_uncommitted_durable_preparation(
                        preparation.preparation_id
                    )
            self._preparations_by_session.pop(session_id, None)
            if preparation is not None:
                self._preparations_by_id.pop(preparation.preparation_id, None)

    def sessions(self) -> list[ConsoleChatSession]:
        """Return native Console sessions in creation order."""
        return list(self._sessions.values())

    def begin_preparation(
        self, preparation: ConsoleTurnPreparation
    ) -> ConsoleTurnPreparation | None:
        """Register one live preparation if the session has no live owner."""

        if not isinstance(preparation, ConsoleTurnPreparation):
            raise TypeError("preparation must be ConsoleTurnPreparation")
        self._session_or_raise(preparation.session_id)
        with self._preparation_lock:
            existing_owner = self._preparations_by_id.get(preparation.preparation_id)
            if existing_owner is not None:
                return existing_owner if existing_owner is preparation else None
            if preparation.preparation_id in self._durable_tombstones:
                return None
            current = self._preparations_by_session.get(preparation.session_id)
            if current is not None and current.state not in {
                ConsoleTurnPreparationState.CANCELLED,
                ConsoleTurnPreparationState.SETTLED,
            }:
                return None
            self._preparations_by_session[preparation.session_id] = preparation
            self._preparations_by_id[preparation.preparation_id] = preparation
            return preparation

    def preparation_for_session(
        self, session_id: str | None
    ) -> ConsoleTurnPreparation | None:
        """Return the immutable preparation currently owned by ``session_id``."""

        if not isinstance(session_id, str) or not session_id:
            return None
        with self._preparation_lock:
            return self._preparations_by_session.get(session_id)

    def preparation_by_id(self, preparation_id: str) -> ConsoleTurnPreparation | None:
        """Return one exact volatile owner, including during session teardown."""

        if not isinstance(preparation_id, str) or not preparation_id:
            return None
        with self._preparation_lock:
            return self._preparations_by_id.get(preparation_id)

    def compare_and_set_preparation(
        self,
        session_id: str,
        transition: ConsolePreparationTransition,
    ) -> ConsoleTurnPreparation | None:
        """Apply one exact preparation transition, returning only a CAS win."""

        if not isinstance(transition, ConsolePreparationTransition):
            raise TypeError("transition must be ConsolePreparationTransition")
        with self._preparation_lock:
            current = self._preparations_by_session.get(session_id)
            if current is None:
                return None
            updated = apply_preparation_transition(current, transition)
            if updated is current:
                return None
            self._preparations_by_session[session_id] = updated
            self._preparations_by_id[updated.preparation_id] = updated
            return updated

    def cancel_preparation(
        self,
        session_id: str,
        preparation_id: str,
        *,
        expected_state: ConsoleTurnPreparationState,
    ) -> ConsoleTurnPreparation | None:
        """Cancel one exact precommit owner and restore its transient inputs."""

        transition = ConsolePreparationTransition(
            preparation_id=preparation_id,
            expected_state=expected_state,
            new_state=ConsoleTurnPreparationState.CANCELLED,
            pause_kind=None,
            new_attempt_id=None,
        )
        with self._preparation_lock:
            current = self._preparations_by_session.get(session_id)
            if current is None:
                return None
            updated = apply_preparation_transition(current, transition)
            if updated is current:
                return None
            self._preparations_by_session[session_id] = updated
            self._preparations_by_id[updated.preparation_id] = updated
            session = self._sessions.get(session_id)
            if session is not None:
                if current.origin == "manual":
                    session.draft = current.executed_draft
                session.title = current.pre_send_title
                session.persisted_conversation_id = current.pre_send_conversation_id
            transient_id = current.transient_user_message_id
            if (
                transient_id is not None
                and self._message_session_index.get(transient_id) == session_id
            ):
                self.delete_message(transient_id)
            return updated

    def remove_preparation(
        self,
        session_id: str,
        preparation_id: str,
        *,
        expected_states: frozenset[ConsoleTurnPreparationState],
    ) -> ConsoleTurnPreparation | None:
        """Remove one exact terminal or abandoned volatile preparation."""

        with self._preparation_lock:
            current = self._preparations_by_session.get(session_id)
            if (
                current is None
                or current.preparation_id != preparation_id
                or current.state not in expected_states
            ):
                return None
            self._preparations_by_session.pop(session_id, None)
            self._preparations_by_id.pop(preparation_id, None)
            return current

    def begin_session_library_destination_attempt(
        self,
        session_id: str,
        authority: ConsoleTurnLibraryAuthority,
        destination: ConsoleResolvedDestination,
        assistant_message_id: str,
    ) -> ConsoleLibraryDestinationRuntimeState:
        """Bind one observed destination to its exact live assistant attempt."""
        if not isinstance(authority, ConsoleTurnLibraryAuthority):
            raise TypeError("authority must be ConsoleTurnLibraryAuthority")
        if not authority.attempt_id:
            raise ValueError("authority.attempt_id must not be empty")
        message = self._message_or_raise(assistant_message_id)
        if self._message_session_index[message.id] != session_id:
            raise ValueError(
                "Assistant message must belong to the destination session."
            )
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Destination attempts must bind to an assistant message.")
        policy = authority.policy
        library_data_possible = (
            policy.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
            or policy.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED
        )
        session = self._session_or_raise(session_id)
        session.library_destination_runtime = (
            update_console_library_destination_runtime(
                session.library_destination_runtime,
                destination,
                library_data_possible=library_data_possible,
            )
        )
        session.library_destination_runtime = replace(
            session.library_destination_runtime,
            owner_attempt_id=authority.attempt_id,
            owner_message_id=assistant_message_id,
        )
        return session.library_destination_runtime

    def settle_session_library_destination(
        self,
        session_id: str,
        *,
        expected_attempt_id: str,
        expected_message_id: str,
    ) -> ConsoleLibraryDestinationRuntimeState:
        """Settle only the exact attempt currently owning live disclosure state."""
        session = self._session_or_raise(session_id)
        runtime = session.library_destination_runtime
        if (
            runtime.owner_attempt_id != expected_attempt_id
            or runtime.owner_message_id != expected_message_id
        ):
            return runtime
        session.library_destination_runtime = (
            settle_console_library_destination_runtime(runtime)
        )
        return session.library_destination_runtime

    def _settle_message_library_destination(
        self,
        session_id: str,
        message_id: str,
    ) -> ConsoleLibraryDestinationRuntimeState:
        """Settle runtime state only when this terminal row owns the attempt."""
        session = self._session_or_raise(session_id)
        runtime = session.library_destination_runtime
        if runtime.owner_message_id != message_id or runtime.owner_attempt_id is None:
            return runtime
        return self.settle_session_library_destination(
            session_id,
            expected_attempt_id=runtime.owner_attempt_id,
            expected_message_id=message_id,
        )

    def set_library_policy_defaults(
        self,
        defaults: ConsoleLibraryPolicyDefaults,
    ) -> None:
        """Replace defaults used only by subsequently created sessions."""
        if not isinstance(defaults, ConsoleLibraryPolicyDefaults):
            raise TypeError("defaults must be ConsoleLibraryPolicyDefaults")
        self._library_policy_defaults = defaults

    async def hydrate_session_library_policy(
        self, session_id: str
    ) -> ConsoleLibraryPolicySnapshot:
        """Hydrate one restored holder off-loop without publishing stale work."""
        session = self._session_or_raise(session_id)
        conversation_id = session.persisted_conversation_id
        coordinator = self.library_policy_coordinator
        if coordinator is None or conversation_id is None:
            session.library_policy_hydrated = True
            return session.library_policy_holder.snapshot
        result = await coordinator.load(session_id, conversation_id)
        current = self._sessions.get(session_id)
        if current is session and current.persisted_conversation_id == conversation_id:
            current.library_policy_hydrated = True
        return result.snapshot

    def has_pending_workspace_projection(self, session_id: str) -> bool:
        """Return whether post-commit workspace projection still needs retry."""
        return session_id in self._pending_workspace_projections

    def retry_pending_workspace_projection(self, session_id: str) -> bool:
        """Reconcile workspace membership from the durable Chat authority."""
        conversation_id = self._pending_workspace_projections.get(session_id)
        if conversation_id is None:
            return True
        project = getattr(self.persistence, "project_workspace_membership", None)
        if not callable(project):
            return False
        try:
            project(conversation_id)
        except Exception:
            logger.bind(
                session_id=session_id,
                conversation_id=conversation_id,
            ).opt(exception=True).warning(
                "Workspace membership projection remains pending."
            )
            return False
        current = self._sessions.get(session_id)
        if current is not None and current.persisted_conversation_id == conversation_id:
            self._pending_workspace_projections.pop(session_id, None)
        return True

    async def reconcile_pending_workspace_projection(self, session_id: str) -> bool:
        """Run a pending registry projection away from the event loop."""
        return await asyncio.to_thread(
            self.retry_pending_workspace_projection,
            session_id,
        )

    def _project_workspace_membership_after_commit(
        self, session: ConsoleChatSession
    ) -> None:
        conversation_id = session.persisted_conversation_id
        if (
            conversation_id is None
            or session.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID
        ):
            self._pending_workspace_projections.pop(session.id, None)
            return
        self._pending_workspace_projections[session.id] = conversation_id
        self.retry_pending_workspace_projection(session.id)

    def stage_session_library_policy(
        self,
        session_id: str,
        candidate: ConsoleLibraryPolicyCandidate,
    ) -> ConsoleLibraryPolicyHolder:
        """Stage an explicit edit without creating a durable conversation."""
        if not isinstance(candidate, ConsoleLibraryPolicyCandidate):
            raise TypeError("candidate must be ConsoleLibraryPolicyCandidate")
        session = self._session_or_raise(session_id)
        current = session.library_policy_holder.snapshot
        session.library_policy_holder.snapshot = ConsoleLibraryPolicySnapshot(
            auto_retrieve=candidate.auto_retrieve,
            assistant_access=candidate.assistant_access,
            policy_revision=current.policy_revision,
            source=current.source,
            error_code=current.error_code,
        )
        session.library_policy_holder.explicitly_staged = True
        self._bump_payload_revision(session_id)
        return session.library_policy_holder

    async def save_session_library_policy(
        self,
        session_id: str,
    ) -> ConsoleLibraryPolicyWriteResult:
        """Persist an explicit policy edit through the shared coordinator."""
        session = self._session_or_raise(session_id)
        coordinator = self.library_policy_coordinator
        if coordinator is None or session.persisted_conversation_id is None:
            raise ValueError("A durable conversation is required for policy save.")
        snapshot = session.library_policy_holder.snapshot
        result = await coordinator.save(
            session_id,
            ConsoleLibraryPolicyCandidate(
                auto_retrieve=snapshot.auto_retrieve,
                assistant_access=snapshot.assistant_access,
            ),
        )
        if result.status.value == "committed":
            session.library_policy_holder.explicitly_staged = False
        return result

    def set_unresolved_promotion_operation(
        self,
        session_id: str,
        operation_id: str | None,
    ) -> None:
        """Expose the narrow unresolved-operation guard needed by later tasks."""
        self._session_or_raise(session_id)
        if operation_id is None:
            self._unresolved_promotion_operations.pop(session_id, None)
            return
        if type(operation_id) is not str or not operation_id.strip():
            raise ValueError("operation_id must be non-empty text or None")
        self._unresolved_promotion_operations[session_id] = operation_id

    def stage_first_persistence(
        self,
        session_id: str,
    ) -> ConsoleStagedConversationIdentity:
        """Reserve first-persistence identity without touching live session state."""
        session = self._session_or_raise(session_id)
        return ConsoleStagedConversationIdentity(
            conversation_id=str(uuid4()),
            title=session.title,
        )

    def stage_durable_turn_identity(
        self,
        session_id: str,
        preparation_id: str,
        *,
        title: str | None = None,
    ) -> ConsoleStagedConversationIdentity:
        """Reserve one stable first-send identity without publishing it."""

        session = self._session_or_raise(session_id)
        if type(preparation_id) is not str or not preparation_id:
            raise ValueError("preparation_id must be non-empty text")
        with self._preparation_lock:
            preparation = self._preparations_by_id.get(preparation_id)
            if preparation is None or preparation.session_id != session_id:
                raise RuntimeError("Durable preparation owner changed.")
            if preparation_id in self._durable_tombstones:
                raise RuntimeError("Durable preparation was already retired.")
            existing = self._durable_identity_by_preparation.get(preparation_id)
            if existing is not None:
                expected_title = title if title is not None else session.title
                expected_conversation_id = (
                    session.persisted_conversation_id or existing.conversation_id
                )
                if (
                    existing.title != expected_title
                    or existing.conversation_id != expected_conversation_id
                ):
                    raise RuntimeError("Durable identity owner changed.")
                return existing
            identity = ConsoleStagedConversationIdentity(
                conversation_id=session.persisted_conversation_id or str(uuid4()),
                title=title if title is not None else session.title,
            )
            self._durable_identity_by_preparation[preparation_id] = identity
            return identity

    def staged_durable_turn_identity_for(
        self, preparation_id: str
    ) -> ConsoleStagedConversationIdentity | None:
        """Return one live staged identity for exact in-process Retry."""

        with self._preparation_lock:
            return self._durable_identity_by_preparation.get(preparation_id)

    def stage_durable_turn_owner_ids(
        self,
        session_id: str,
        preparation_id: str,
        *,
        user_message_id: str,
        assistant_message_id: str | None = None,
    ) -> _ConsoleStagedDurableOwnerIds:
        """Reserve exact message owners once and reuse them across persistence Retry."""

        if type(user_message_id) is not str or not user_message_id:
            raise ValueError("user_message_id must be non-empty text")
        if assistant_message_id is not None and (
            type(assistant_message_id) is not str or not assistant_message_id
        ):
            raise ValueError("assistant_message_id must be non-empty text or None")
        with self._preparation_lock:
            preparation = self._preparations_by_id.get(preparation_id)
            if preparation is None or preparation.session_id != session_id:
                raise RuntimeError("Durable preparation owner changed.")
            if preparation_id in self._durable_tombstones:
                raise RuntimeError("Durable preparation was already retired.")
            existing = self._durable_owner_ids_by_preparation.get(preparation_id)
            if existing is not None:
                if existing.user_message_id != user_message_id or (
                    assistant_message_id is not None
                    and existing.assistant_message_id != assistant_message_id
                ):
                    raise RuntimeError("Durable message owner changed.")
                return existing
            owners = _ConsoleStagedDurableOwnerIds(
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id or str(uuid4()),
            )
            self._durable_owner_ids_by_preparation[preparation_id] = owners
            return owners

    @staticmethod
    def _canonical_fingerprint_value(value: object, *, depth: int = 0) -> object:
        """Return deterministic JSON-safe data without retaining binary bodies."""

        if depth > 12:
            raise ValueError("Durable acceptance fingerprint is not canonical.")
        if value is None or type(value) in {bool, int, str}:
            return value
        if isinstance(value, Enum):
            return {
                "enum": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": value.value,
            }
        if type(value) is bytes:
            return {
                "bytes_sha256": hashlib.sha256(value).hexdigest(),
                "bytes_length": len(value),
            }
        if isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                raise TypeError("Durable acceptance fingerprint keys must be text.")
            return {
                key: ConsoleChatStore._canonical_fingerprint_value(
                    item, depth=depth + 1
                )
                for key, item in sorted(value.items())
            }
        if type(value) in {tuple, list}:
            return [
                ConsoleChatStore._canonical_fingerprint_value(item, depth=depth + 1)
                for item in value
            ]
        if is_dataclass(value) and not isinstance(value, type):
            parameters = getattr(type(value), "__dataclass_params__", None)
            if parameters is None or not parameters.frozen:
                raise TypeError(
                    "Durable acceptance fingerprint requires a frozen dataclass."
                )
            return {
                "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {
                    item.name: ConsoleChatStore._canonical_fingerprint_value(
                        getattr(value, item.name), depth=depth + 1
                    )
                    for item in fields(value)
                },
            }
        raise TypeError("Durable acceptance fingerprint input is not canonical.")

    @classmethod
    def _contribution_fingerprint(cls, contribution: object) -> object:
        provider = getattr(contribution, "durable_acceptance_fingerprint", None)
        if callable(provider):
            payload = provider()
        elif is_dataclass(contribution) and not isinstance(contribution, type):
            parameters = getattr(type(contribution), "__dataclass_params__", None)
            if parameters is None or not parameters.frozen:
                raise TypeError(
                    "Durable contribution fingerprint requires a frozen dataclass "
                    "or durable_acceptance_fingerprint()."
                )
            payload = contribution
        else:
            raise TypeError("Durable contribution fingerprint requires canonical data.")
        canonical = {
            "type": f"{type(contribution).__module__}.{type(contribution).__qualname__}",
            "payload": cls._canonical_fingerprint_value(payload),
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 16_384:
            raise ValueError("Durable contribution fingerprint exceeds its bound.")
        return canonical

    @classmethod
    def _durable_acceptance_fingerprint(
        cls,
        acceptance: ConsoleDurableTurnAcceptance,
        preparation: ConsoleTurnPreparation,
        identity: ConsoleStagedConversationIdentity,
        owners: _ConsoleStagedDurableOwnerIds,
        policy_candidate: ConsoleLibraryPolicyCandidate,
        conversation_kwargs: Mapping[str, object],
    ) -> ConsoleDurableAcceptanceFingerprint:
        plan = {
            "preparation_id": acceptance.preparation_id,
            "attempt_id": acceptance.attempt_id,
            "session_id": preparation.session_id,
            "effect_owner": {
                "session_id": preparation.session_id,
                "preparation_id": acceptance.preparation_id,
                "assistant_message_id": owners.assistant_message_id,
            },
            "identity": {
                "conversation_id": identity.conversation_id,
                "title_sha256": hashlib.sha256(
                    identity.title.encode("utf-8")
                ).hexdigest(),
                "user_message_id": owners.user_message_id,
                "assistant_message_id": owners.assistant_message_id,
            },
            "policy_candidate": cls._canonical_fingerprint_value(policy_candidate),
            "conversation_kwargs": cls._canonical_fingerprint_value(
                conversation_kwargs
            ),
            "user_content_sha256": hashlib.sha256(
                acceptance.user_content.encode("utf-8")
            ).hexdigest(),
            "parent_message_id": acceptance.parent_message_id,
            "attachments": cls._canonical_fingerprint_value(acceptance.attachments),
            "origin": acceptance.origin,
            "queue_entry_id": acceptance.queue_entry_id,
            "frozen_authority": cls._canonical_fingerprint_value(
                acceptance.frozen_authority
            ),
            "resolved_destination": cls._canonical_fingerprint_value(
                acceptance.resolved_destination
            ),
            "reconstructability": cls._canonical_fingerprint_value(
                acceptance.reconstructability
            ),
            "contributions": [
                cls._contribution_fingerprint(contribution)
                for contribution in acceptance.contributions
            ],
        }
        encoded = json.dumps(
            plan,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return ConsoleDurableAcceptanceFingerprint(
            preparation_id=acceptance.preparation_id,
            session_id=preparation.session_id,
            conversation_id=acceptance.conversation_id,
            title_hash=hashlib.sha256(identity.title.encode("utf-8")).hexdigest(),
            attempt_id=acceptance.attempt_id,
            origin=acceptance.origin,
            queue_entry_id=acceptance.queue_entry_id,
            user_message_id=owners.user_message_id,
            assistant_message_id=owners.assistant_message_id,
            digest=hashlib.sha256(encoded).hexdigest(),
        )

    def durable_acceptance_fingerprint_for(
        self, preparation_id: str
    ) -> ConsoleDurableAcceptanceFingerprint | None:
        """Return the body-free live or bounded retired owner fingerprint."""

        with self._preparation_lock:
            current = self._durable_fingerprint_by_preparation.get(preparation_id)
            if current is not None:
                return current
            tombstone = self._durable_tombstones.get(preparation_id)
            return tombstone.fingerprint if tombstone is not None else None

    def validate_durable_acceptance_fingerprint(
        self, fingerprint: ConsoleDurableAcceptanceFingerprint
    ) -> None:
        """Fail closed unless ``fingerprint`` is the exact registered owner."""

        if not isinstance(fingerprint, ConsoleDurableAcceptanceFingerprint):
            raise TypeError("fingerprint must be ConsoleDurableAcceptanceFingerprint")
        current = self.durable_acceptance_fingerprint_for(fingerprint.preparation_id)
        if current != fingerprint:
            raise RuntimeError("Durable acceptance fingerprint changed.")

    @property
    def durable_preparation_lock(self) -> threading.RLock:
        """Return the single lock guarding preparation and durable caches."""

        return self._preparation_lock

    def session_library_policy_candidate(
        self, session_id: str
    ) -> ConsoleLibraryPolicyCandidate:
        """Return the exact policy values staged on one native session."""

        snapshot = self._session_or_raise(session_id).library_policy_holder.snapshot
        return ConsoleLibraryPolicyCandidate(
            auto_retrieve=snapshot.auto_retrieve,
            assistant_access=snapshot.assistant_access,
        )

    def _pause_failed_durable_commit_locked(
        self,
        *,
        reservation: _ConsoleDurableCommitReservation,
    ) -> bool:
        """Atomically release and pause the exact failed commit owner.

        The caller holds ``_preparation_lock``.  This method performs no
        callback or persistence work; it only validates and replaces the exact
        in-memory owner guarded by that lock.
        """

        preparation_id = reservation.preparation_id
        if self._durable_commit_in_flight.get(preparation_id) is not reservation:
            return False
        preparation = self._preparations_by_id.get(preparation_id)
        if (
            preparation is None
            or self._preparations_by_session.get(reservation.session_id)
            is not preparation
            or reservation.session_id not in self._sessions
            or preparation.session_id != reservation.session_id
            or preparation.attempt_id != reservation.attempt_id
            or preparation.origin != reservation.origin
            or preparation.queue_entry_id != reservation.queue_entry_id
            or preparation.state is not ConsoleTurnPreparationState.COMMITTING
        ):
            return False
        identity = self._durable_identity_by_preparation.get(preparation_id)
        owners = self._durable_owner_ids_by_preparation.get(preparation_id)
        if (
            identity is None
            or identity.conversation_id != reservation.conversation_id
            or owners is None
            or owners.user_message_id != reservation.user_message_id
            or owners.assistant_message_id != reservation.assistant_message_id
        ):
            return False
        paused = apply_preparation_transition(
            preparation,
            ConsolePreparationTransition(
                preparation_id=preparation_id,
                expected_state=ConsoleTurnPreparationState.COMMITTING,
                new_state=ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.PERSISTENCE,
                new_attempt_id=None,
            ),
        )
        if paused is preparation:
            return False
        self._durable_commit_in_flight.pop(preparation_id)
        self._preparations_by_session[reservation.session_id] = paused
        self._preparations_by_id[preparation_id] = paused
        return True

    def commit_durable_turn(
        self, acceptance: ConsoleDurableTurnAcceptance
    ) -> ConsoleDurableTurnCommit:
        """Commit one durable turn without publishing any live owner state."""

        if not isinstance(acceptance, ConsoleDurableTurnAcceptance):
            raise TypeError("acceptance must be ConsoleDurableTurnAcceptance")
        reservation: _ConsoleDurableCommitReservation | None = None
        fingerprint: ConsoleDurableAcceptanceFingerprint | None = None
        try:
            with self._preparation_lock:
                preparation = self._preparations_by_id.get(acceptance.preparation_id)
                if preparation is None or preparation.session_id not in self._sessions:
                    raise RuntimeError("Durable preparation is unavailable.")
                session = self._sessions[preparation.session_id]
                existing_reservation = self._durable_commit_in_flight.get(
                    acceptance.preparation_id
                )
                if existing_reservation is not None:
                    staged_identity = self._durable_identity_by_preparation.get(
                        acceptance.preparation_id
                    )
                    staged_owners = self._durable_owner_ids_by_preparation.get(
                        acceptance.preparation_id
                    )
                    if (
                        existing_reservation.session_id != preparation.session_id
                        or existing_reservation.attempt_id != acceptance.attempt_id
                        or existing_reservation.conversation_id
                        != acceptance.conversation_id
                        or existing_reservation.user_message_id
                        != acceptance.user_message_id
                        or existing_reservation.assistant_message_id
                        != acceptance.assistant_message_id
                        or existing_reservation.origin != acceptance.origin
                        or existing_reservation.queue_entry_id
                        != acceptance.queue_entry_id
                        or staged_identity is None
                        or staged_identity.conversation_id
                        != existing_reservation.conversation_id
                        or staged_owners is None
                        or staged_owners.user_message_id
                        != existing_reservation.user_message_id
                        or staged_owners.assistant_message_id
                        != existing_reservation.assistant_message_id
                    ):
                        raise RuntimeError("Durable message owner changed.")
                    raise RuntimeError(
                        "Durable acceptance commit is already in flight."
                    )
                staged = self._durable_identity_by_preparation.get(
                    acceptance.preparation_id
                )
                identity = self.stage_durable_turn_identity(
                    session.id,
                    acceptance.preparation_id,
                    title=staged.title if staged is not None else session.title,
                )
                owners = self.stage_durable_turn_owner_ids(
                    session.id,
                    acceptance.preparation_id,
                    user_message_id=acceptance.user_message_id,
                    assistant_message_id=acceptance.assistant_message_id,
                )
                if (
                    preparation.attempt_id != acceptance.attempt_id
                    or preparation.origin != acceptance.origin
                    or preparation.queue_entry_id != acceptance.queue_entry_id
                    or identity.conversation_id != acceptance.conversation_id
                ):
                    raise RuntimeError("Durable acceptance identity changed.")
                if preparation.state is not ConsoleTurnPreparationState.COMMITTING:
                    raise RuntimeError("Durable preparation is not committing.")
                scope_type, workspace_id = self._persistence_scope(session)
                local_character_id = session.local_character_id()
                conversation_kwargs: dict[str, object] = {
                    "conversation_title": identity.title,
                    "workspace_id": workspace_id,
                    "scope_type": scope_type,
                    "system_prompt": (
                        session.settings.system_prompt
                        if session.settings is not None
                        else None
                    ),
                    "runtime_backend": session.runtime_backend,
                    "assistant_kind": session.assistant_kind,
                    "assistant_id": session.assistant_id,
                    "assistant_authority_id": session.assistant_authority_id,
                    "character_id": local_character_id,
                    "character_name": (
                        session.character_name
                        if local_character_id is not None
                        else None
                    ),
                }
                if session.speech_preferences != ConsoleSpeechPreferences():
                    conversation_kwargs["speech_preferences"] = (
                        session.speech_preferences
                    )
                policy_candidate = self.session_library_policy_candidate(session.id)
                if acceptance.preparation_id not in self._durable_commit_by_preparation:
                    reservation = _ConsoleDurableCommitReservation(
                        caller_token=object(),
                        owner_thread_id=threading.get_ident(),
                        preparation_id=acceptance.preparation_id,
                        attempt_id=acceptance.attempt_id,
                        session_id=session.id,
                        conversation_id=identity.conversation_id,
                        user_message_id=owners.user_message_id,
                        assistant_message_id=owners.assistant_message_id,
                        origin=acceptance.origin,
                        queue_entry_id=acceptance.queue_entry_id,
                    )
                    self._durable_commit_in_flight[acceptance.preparation_id] = (
                        reservation
                    )
            fingerprint = self._durable_acceptance_fingerprint(
                acceptance,
                preparation,
                identity,
                owners,
                policy_candidate,
                conversation_kwargs,
            )
            with self._preparation_lock:
                if reservation is None:
                    existing_fingerprint = self._durable_fingerprint_by_preparation.get(
                        acceptance.preparation_id
                    )
                    existing_commit = self._durable_commit_by_preparation.get(
                        acceptance.preparation_id
                    )
                    if existing_fingerprint != fingerprint:
                        raise RuntimeError("Durable acceptance fingerprint changed.")
                    if existing_commit is None:
                        raise RuntimeError("Durable acceptance is unavailable.")
                    return existing_commit
                if (
                    self._durable_commit_in_flight.get(acceptance.preparation_id)
                    is not reservation
                ):
                    raise RuntimeError("Durable commit reservation changed.")
                current_preparation = self._preparations_by_id.get(
                    acceptance.preparation_id
                )
                current_identity = self._durable_identity_by_preparation.get(
                    acceptance.preparation_id
                )
                current_owners = self._durable_owner_ids_by_preparation.get(
                    acceptance.preparation_id
                )
                if (
                    current_preparation is None
                    or current_preparation.session_id != reservation.session_id
                    or current_preparation.attempt_id != reservation.attempt_id
                    or current_identity is not identity
                    or current_identity.conversation_id != reservation.conversation_id
                    or current_owners is not owners
                    or current_owners.user_message_id != reservation.user_message_id
                    or current_owners.assistant_message_id
                    != reservation.assistant_message_id
                ):
                    raise RuntimeError("Durable commit reservation owner changed.")
                retired = self._durable_tombstones.get(acceptance.preparation_id)
                if retired is not None:
                    if retired.fingerprint != fingerprint:
                        raise RuntimeError("Durable acceptance fingerprint changed.")
                    raise RuntimeError("Durable acceptance was already retired.")
                existing_fingerprint = self._durable_fingerprint_by_preparation.get(
                    acceptance.preparation_id
                )
                if (
                    existing_fingerprint is not None
                    and existing_fingerprint != fingerprint
                ):
                    raise RuntimeError("Durable acceptance fingerprint changed.")
                existing_commit = self._durable_commit_by_preparation.get(
                    acceptance.preparation_id
                )
                if existing_commit is not None:
                    raise RuntimeError("Durable commit reservation was superseded.")
                self._durable_fingerprint_by_preparation[acceptance.preparation_id] = (
                    fingerprint
                )
            durable_commit = getattr(self.persistence, "commit_durable_turn", None)
            if not callable(durable_commit):
                raise RuntimeError("Durable Console persistence is unavailable.")
            checkpoint = durable_commit(
                acceptance=acceptance,
                policy_candidate=policy_candidate,
                conversation_kwargs=conversation_kwargs,
            )
            commit = ConsoleDurableTurnCommit(
                identity=identity,
                user_message_id=owners.user_message_id,
                user_message_version=checkpoint.user_message_version,
                assistant_message_id=owners.assistant_message_id,
                assistant_message_version=checkpoint.assistant_message_version,
                checkpoint=checkpoint,
            )
            with self._preparation_lock:
                if (
                    self._durable_commit_in_flight.get(acceptance.preparation_id)
                    is not reservation
                ):
                    raise RuntimeError("Durable commit reservation changed.")
                if (
                    self._durable_fingerprint_by_preparation.get(
                        acceptance.preparation_id
                    )
                    != fingerprint
                ):
                    raise RuntimeError("Durable acceptance fingerprint changed.")
                self._durable_commit_in_flight.pop(acceptance.preparation_id, None)
                self._durable_commit_by_preparation[acceptance.preparation_id] = commit
                self.begin_durable_postcommit_effects(
                    preparation_id=acceptance.preparation_id,
                    session_id=session.id,
                    assistant_message_id=owners.assistant_message_id,
                    fingerprint=fingerprint,
                )
            return commit
        except Exception:
            if reservation is None:
                raise
            with self._preparation_lock:
                owner_paused = self._pause_failed_durable_commit_locked(
                    reservation=reservation,
                )
            if not owner_paused:
                raise
            logger.bind(
                session_id=session.id,
                preparation_id=preparation.preparation_id,
            ).warning("console_durable_turn_commit_failed")
            raise

    def begin_durable_postcommit_effects(
        self,
        *,
        preparation_id: str,
        session_id: str,
        assistant_message_id: str,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> ConsoleDurablePostcommitEffects:
        """Create or return one preparation-keyed postcommit ledger."""

        self._session_or_raise(session_id)
        with self._preparation_lock:
            self._require_durable_fingerprint_locked(preparation_id, fingerprint)
            self._durable_active_postcommit.add(preparation_id)
            existing = self._durable_effects_by_preparation.get(preparation_id)
            if existing is not None:
                if (
                    existing.session_id != session_id
                    or existing.assistant_message_id != assistant_message_id
                ):
                    raise RuntimeError("Durable postcommit owner changed.")
                return existing
            effects = ConsoleDurablePostcommitEffects(
                preparation_id=preparation_id,
                session_id=session_id,
                assistant_message_id=assistant_message_id,
            )
            self._durable_effects_by_preparation[preparation_id] = effects
            return effects

    def durable_postcommit_effects_for(
        self,
        preparation_id: str | None,
        *,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> ConsoleDurablePostcommitEffects | None:
        """Return the immutable effect ledger for one committed turn."""

        if not preparation_id:
            return None
        with self._preparation_lock:
            self._require_durable_fingerprint_locked(preparation_id, fingerprint)
            return self._durable_effects_by_preparation.get(preparation_id)

    def durable_completed_effects_for(
        self,
        preparation_id: str,
        *,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> frozenset[str]:
        """Return completed effect names from the live ledger OR a tombstone.

        TASK-22587: recovery needs to know whether the checkpoint transition
        ran, and it asks *after* a failure -- by which point the user may have
        closed the chat and retired the preparation. Reading the ledger
        directly raises there, which masked the original failure. The tombstone
        retains `completed` for exactly this reason, so the answer survives a
        close instead of becoming an exception.
        """

        with self._preparation_lock:
            if self._durable_retired_locked(preparation_id, fingerprint):
                tombstone = self._durable_tombstones.get(preparation_id)
                return tombstone.completed if tombstone is not None else frozenset()
            self._require_durable_fingerprint_locked(preparation_id, fingerprint)
            effects = self._durable_effects_by_preparation.get(preparation_id)
            return effects.completed if effects is not None else frozenset()

    def durable_turn_commit_for(
        self,
        preparation_id: str,
        *,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> ConsoleDurableTurnCommit | None:
        """Return one app-lifetime durable acceptance result."""

        with self._preparation_lock:
            self._require_durable_fingerprint_locked(preparation_id, fingerprint)
            return self._durable_commit_by_preparation.get(preparation_id)

    def complete_durable_postcommit_effect(
        self,
        preparation_id: str,
        effect_name: str,
        *,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> ConsoleDurablePostcommitEffects:
        """Mark one effect complete only after its caller reports success."""

        with self._preparation_lock:
            self._require_durable_fingerprint_locked(preparation_id, fingerprint)
            current = self._durable_effects_by_preparation.get(preparation_id)
            if current is None:
                raise RuntimeError("Durable postcommit ledger is unavailable.")
            updated = replace(
                current,
                completed=current.completed | {effect_name},
            )
            self._durable_effects_by_preparation[preparation_id] = updated
            self._durable_effects_in_flight.discard((preparation_id, effect_name))
            return updated

    def claim_durable_postcommit_effect(
        self,
        preparation_id: str,
        effect_name: str,
        *,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> bool:
        """Claim one incomplete effect so concurrent re-entry cannot duplicate it."""

        key = (preparation_id, effect_name)
        with self._preparation_lock:
            self._require_durable_fingerprint_locked(preparation_id, fingerprint)
            current = self._durable_effects_by_preparation.get(preparation_id)
            if (
                current is None
                or effect_name in current.completed
                or key in self._durable_effects_in_flight
            ):
                return False
            self._durable_effects_in_flight.add(key)
            return True

    def abandon_durable_postcommit_effect(
        self,
        preparation_id: str,
        effect_name: str,
        *,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> None:
        """Release a failed effect claim without recording completion."""

        with self._preparation_lock:
            if self._durable_retired_locked(preparation_id, fingerprint):
                # TASK-22587: `retire_durable_acceptance` already dropped every
                # in-flight key for this preparation, so there is nothing left
                # to release and nothing left to protect. Raising here would
                # only mask the failure that sent us down the release path.
                return
            self._require_durable_fingerprint_locked(preparation_id, fingerprint)
            self._durable_effects_in_flight.discard((preparation_id, effect_name))

    def _require_durable_fingerprint_locked(
        self,
        preparation_id: str,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> None:
        if not isinstance(fingerprint, ConsoleDurableAcceptanceFingerprint):
            raise TypeError("fingerprint must be ConsoleDurableAcceptanceFingerprint")
        if self._durable_fingerprint_by_preparation.get(preparation_id) != fingerprint:
            if self._durable_retired_locked(preparation_id, fingerprint):
                raise ConsoleDurableAcceptanceRetired(
                    "Durable acceptance was retired."
                )
            raise RuntimeError("Durable postcommit fingerprint changed.")

    def _durable_retired_locked(
        self,
        preparation_id: str,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> bool:
        """True when this exact preparation was retired, not mutated."""

        tombstone = self._durable_tombstones.get(preparation_id)
        return tombstone is not None and tombstone.fingerprint == fingerprint

    def retire_durable_acceptance(
        self,
        preparation_id: str,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> None:
        """Drop content-bearing caches and retain one bounded owner tombstone."""

        with self._preparation_lock:
            current = self._durable_fingerprint_by_preparation.get(preparation_id)
            if current != fingerprint:
                if self._durable_retired_locked(preparation_id, fingerprint):
                    # TASK-22587: closing a chat retires the preparation, and
                    # the postcommit sequence retires it again when it ends.
                    # Retiring what is already retired is a no-op, not a bug --
                    # the tombstone proves this is the SAME acceptance.
                    return
                raise RuntimeError("Durable acceptance fingerprint changed.")
            effects = self._durable_effects_by_preparation.get(preparation_id)
            completed = effects.completed if effects is not None else frozenset()
            self._durable_identity_by_preparation.pop(preparation_id, None)
            self._durable_owner_ids_by_preparation.pop(preparation_id, None)
            self._durable_commit_by_preparation.pop(preparation_id, None)
            self._durable_effects_by_preparation.pop(preparation_id, None)
            self._durable_fingerprint_by_preparation.pop(preparation_id, None)
            self._durable_commit_in_flight.pop(preparation_id, None)
            self._durable_effects_in_flight = {
                key
                for key in self._durable_effects_in_flight
                if key[0] != preparation_id
            }
            self._durable_tombstones.pop(preparation_id, None)
            self._durable_tombstones[preparation_id] = _ConsoleDurableTombstone(
                fingerprint=fingerprint,
                completed=completed,
            )
            self._evict_durable_tombstones_locked()

    def _evict_durable_tombstones_locked(self) -> None:
        """Hold the cap while preserving retirement proof still in use.

        TASK-22587 (Qodo review of #2123): a plain FIFO `popitem` could reclaim
        the tombstone of a preparation whose postcommit sequence was STILL
        RUNNING, and that tombstone is the only proof its retirement was an
        ordinary close rather than a mutation. Losing it put the generic
        fingerprint-change error back on the in-flight effect -- making
        correctness depend on unrelated session-close volume.

        Protected entries are skipped, oldest-first, so the cap still holds:
        if every entry is protected the oldest is evicted anyway, because a
        bounded cache that can be pinned open without limit is a leak.
        """

        while len(self._durable_tombstones) > self.DURABLE_TOMBSTONE_CAP:
            victim = next(
                (
                    key
                    for key in self._durable_tombstones
                    if key not in self._durable_active_postcommit
                ),
                None,
            )
            if victim is None:
                self._durable_tombstones.popitem(last=False)
                continue
            self._durable_tombstones.pop(victim, None)

    def durable_acceptance_retired(
        self,
        preparation_id: str,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> bool:
        """Return whether THIS acceptance was retired, rather than mutated.

        The public form of the tombstone check TASK-22587 introduced. Callers
        outside the store need it to tell "the user closed the chat" from "the
        owner changed underneath me", which are otherwise indistinguishable
        once the live fingerprint is gone.

        Args:
            preparation_id: The preparation whose retirement is in question.
            fingerprint: The acceptance the caller believes it is settling.
                Matching matters: a tombstone under this id for a DIFFERENT
                acceptance is an owner change, not a close.

        Returns:
            True when a tombstone exists for ``preparation_id`` and carries
            exactly ``fingerprint``; False otherwise, including when no
            tombstone exists at all.
        """

        with self._preparation_lock:
            return self._durable_retired_locked(preparation_id, fingerprint)

    def release_durable_postcommit_activity(self, preparation_id: str) -> None:
        """Allow this preparation's tombstone to be evicted again.

        Called once the postcommit sequence is finished with the preparation,
        by either the normal tail or the closed-session path (TASK-22587).
        """

        with self._preparation_lock:
            self._durable_active_postcommit.discard(preparation_id)

    def discard_uncommitted_durable_preparation(self, preparation_id: str) -> None:
        """Forget staged content for an acceptance which never committed."""

        with self._preparation_lock:
            if preparation_id in self._durable_commit_by_preparation:
                raise RuntimeError("Committed durable acceptance cannot be discarded.")
            self._durable_identity_by_preparation.pop(preparation_id, None)
            self._durable_owner_ids_by_preparation.pop(preparation_id, None)
            self._durable_fingerprint_by_preparation.pop(preparation_id, None)
            self._durable_commit_in_flight.pop(preparation_id, None)

    def durable_content_retention_count(self) -> int:
        """Return the number of content-bearing durable recovery cache entries."""

        with self._preparation_lock:
            return len(self._durable_identity_by_preparation) + len(
                self._durable_commit_by_preparation
            )

    def durable_tombstone_count(self) -> int:
        """Return the current bounded durable owner tombstone count."""

        with self._preparation_lock:
            return len(self._durable_tombstones)

    def durable_retention_debug_snapshot(self) -> tuple[object, ...]:
        """Return a body-free retention projection for privacy verification."""

        with self._preparation_lock:
            return tuple(self._durable_tombstones.values())

    def publish_durable_turn_identity(
        self, session_id: str, commit: ConsoleDurableTurnCommit
    ) -> None:
        """Publish committed identity and the matching durable policy snapshot."""

        self.publish_committed_identity(session_id, commit.identity)
        session = self._session_or_raise(session_id)
        repository = getattr(
            self.persistence, "console_library_policy_repository", None
        )
        if repository is None:
            raise RuntimeError("Durable Console Library policy is unavailable.")
        result = repository.read(commit.identity.conversation_id)
        if result.durable_policy is None:
            raise RuntimeError("Committed Console Library policy is unavailable.")
        session.library_policy_holder.snapshot = result.snapshot
        session.library_policy_holder.explicitly_staged = False
        if self.library_policy_coordinator is not None:
            self.library_policy_coordinator.register_holder(
                session.id,
                commit.identity.conversation_id,
                session.library_policy_holder,
            )

    def publish_durable_turn_owners(
        self,
        session_id: str,
        commit: ConsoleDurableTurnCommit,
        *,
        terminal_citation_finalizer: TerminalCitationFinalizer | None = None,
        defer_terminal_persistence: bool = False,
    ) -> tuple[ConsoleChatMessage, ConsoleChatMessage]:
        """Hydrate the already-committed USER and assistant live owners."""

        user, assistant = self._hydrate_durable_turn_owner_messages(
            session_id,
            commit,
            terminal_citation_finalizer=terminal_citation_finalizer,
            defer_terminal_persistence=defer_terminal_persistence,
        )
        self.publish_durable_dispatch_checkpoint(
            session_id,
            commit.checkpoint,
            in_flight=False,
        )
        return self._snapshot(user), self._snapshot(assistant)

    def publish_durable_recovery_owner(
        self,
        session_id: str,
        commit: ConsoleDurableTurnCommit,
        *,
        terminal_citation_finalizer: TerminalCitationFinalizer | None = None,
        defer_terminal_persistence: bool = False,
    ) -> tuple[ConsoleChatMessage, ConsoleChatMessage]:
        """Expose one committed owner without completing a postcommit effect."""

        user, assistant = self._hydrate_durable_turn_owner_messages(
            session_id,
            commit,
            terminal_citation_finalizer=terminal_citation_finalizer,
            defer_terminal_persistence=defer_terminal_persistence,
        )
        self.publish_durable_dispatch_checkpoint(
            session_id,
            commit.checkpoint,
            in_flight=False,
        )
        self.mark_dispatch_recovery_needed(session_id, commit.assistant_message_id)
        return self._snapshot(user), self._snapshot(assistant)

    def _hydrate_durable_turn_owner_messages(
        self,
        session_id: str,
        commit: ConsoleDurableTurnCommit,
        *,
        terminal_citation_finalizer: TerminalCitationFinalizer | None,
        defer_terminal_persistence: bool,
    ) -> tuple[ConsoleChatMessage, ConsoleChatMessage]:
        """Hydrate exact committed messages without changing the effect ledger."""

        user = self._message_or_raise(commit.user_message_id)
        if self._message_session_index.get(user.id) != session_id:
            raise RuntimeError("Committed USER owner changed sessions.")
        user.persisted_message_id = commit.user_message_id
        try:
            assistant = self._message_or_raise(commit.assistant_message_id)
        except KeyError:
            self.append_message(
                session_id,
                role=ConsoleMessageRole.ASSISTANT,
                content="",
                persist=False,
                terminal_citation_finalizer=terminal_citation_finalizer,
                defer_terminal_persistence=defer_terminal_persistence,
                message_id=commit.assistant_message_id,
            )
            assistant = self._message_or_raise(commit.assistant_message_id)
        if assistant.role is not ConsoleMessageRole.ASSISTANT:
            raise RuntimeError("Committed assistant owner changed role.")
        assistant.persisted_message_id = commit.assistant_message_id
        # TASK-22302: arm here, AFTER the durable id is assigned.
        # `append_message` gates arming on its own `persist` flag, which is
        # False on this path and correctly so -- the checkpoint already wrote
        # the row. But "this call does not create the row" is not "this message
        # has no durable row", and arming needs the latter.
        if (
            terminal_citation_finalizer is not None
            and self._citation_persistence_ready()
        ):
            self._terminal_citation_finalizers[assistant.id] = (
                terminal_citation_finalizer
            )
            self._terminal_persistence_deferred_ids.add(assistant.id)
        return user, assistant

    def publish_committed_identity(
        self,
        session_id: str,
        identity: ConsoleStagedConversationIdentity,
    ) -> None:
        """Publish identity only after its caller-owned transaction exits."""
        if not isinstance(identity, ConsoleStagedConversationIdentity):
            raise TypeError("identity must be ConsoleStagedConversationIdentity")
        session = self._session_or_raise(session_id)
        session.persisted_conversation_id = identity.conversation_id
        session.title = identity.title
        self._flush_staged_capture_policy(session)

    def session_settings(self, session_id: str) -> ConsoleSessionSettings | None:
        """Return in-memory settings for a native Console session."""
        return self._session_or_raise(session_id).settings

    def session_context_policy_overrides(
        self, session_id: str
    ) -> ConsoleContextPolicyOverrides:
        """Return sparse conversation-owned context-policy overrides."""
        return self._session_or_raise(session_id).context_policy_overrides

    def set_session_context_policy_overrides(
        self,
        session_id: str,
        overrides: ConsoleContextPolicyOverrides,
    ) -> tuple[ConsoleChatSession, bool]:
        """Stage policy and write through only when a conversation exists.

        Returns an honest ``(session, persisted)`` pair. Applying policy to an
        empty tab never calls ``persist_session_if_needed`` and therefore
        cannot create an empty conversation row.
        """
        if not isinstance(overrides, ConsoleContextPolicyOverrides):
            raise TypeError("overrides must be ConsoleContextPolicyOverrides")
        session = self._session_or_raise(session_id)
        session.context_policy_overrides = overrides
        session.context_policy_error = None
        self._bump_payload_revision(session_id)
        if session.persisted_conversation_id is None or self.persistence is None:
            return session, True
        writer = getattr(self.persistence, "update_conversation_context_policy", None)
        if not callable(writer):
            return session, False
        try:
            writer(
                conversation_id=session.persisted_conversation_id,
                overrides=overrides,
            )
        except Exception:
            logger.error(
                "Failed to persist Console context policy; in-memory policy "
                "keeps the applied value."
            )
            return session, False
        return session, True

    def set_auto_speak(
        self, session_id: str, enabled: bool
    ) -> tuple[ConsoleChatSession, bool]:
        """Enable or disable automatic reply speech for one conversation."""
        if type(enabled) is not bool:
            raise ValueError("enabled must be an exact boolean.")
        session = self._session_or_raise(session_id)
        return self._set_speech_preferences(
            session,
            replace(session.speech_preferences, auto_speak=enabled),
        )

    def speech_preference_epoch(self, session_id: str) -> int:
        """Return the process-local revision of one session's speech opt-in."""
        self._session_or_raise(session_id)
        return self._speech_preference_epochs.get(session_id, 0)

    def _bump_speech_preference_epoch(self, session_id: str) -> None:
        self._speech_preference_epoch_sequence += 1
        self._speech_preference_epochs[session_id] = (
            self._speech_preference_epoch_sequence
        )

    def pause_auto_speak(self, session_id: str) -> tuple[ConsoleChatSession, bool]:
        """Persistently pause automatic reply speech for one conversation."""
        session = self._session_or_raise(session_id)
        return self._set_speech_preferences(
            session,
            replace(session.speech_preferences, paused=True),
        )

    def resume_auto_speak(self, session_id: str) -> tuple[ConsoleChatSession, bool]:
        """Resume automatic reply speech for one conversation."""
        session = self._session_or_raise(session_id)
        return self._set_speech_preferences(
            session,
            replace(session.speech_preferences, paused=False),
        )

    def confirm_auto_speak_destination(
        self, session_id: str, destination: str
    ) -> tuple[ConsoleChatSession, bool]:
        """Record consent for one canonical TTS destination fingerprint."""
        session = self._session_or_raise(session_id)
        return self._set_speech_preferences(
            session,
            replace(session.speech_preferences, consent_destination=destination),
        )

    def _set_speech_preferences(
        self,
        session: ConsoleChatSession,
        preferences: ConsoleSpeechPreferences,
    ) -> tuple[ConsoleChatSession, bool]:
        """Apply speech state after its versioned durable write succeeds."""
        if session.persisted_conversation_id is None:
            if preferences == session.speech_preferences:
                return session, True
            session.speech_preferences = preferences
            session.updated_at = _utc_now_iso()
            self._bump_speech_preference_epoch(session.id)
            return session, True
        if self.persistence is None:
            return session, False
        version_reader = getattr(self.persistence, "get_conversation_version", None)
        writer = getattr(
            self.persistence,
            "update_conversation_speech_preferences",
            None,
        )
        if not callable(version_reader) or not callable(writer):
            return session, False
        try:
            expected_version = version_reader(session.persisted_conversation_id)
            if type(expected_version) is not int or expected_version < 1:
                return session, False
            persisted = bool(
                writer(
                    conversation_id=session.persisted_conversation_id,
                    preferences=preferences,
                    expected_version=expected_version,
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist Console reply-speech preferences "
                "(exception_type={}).",
                type(exc).__name__,
            )
            return session, False
        if not persisted:
            return session, False
        changed = preferences != session.speech_preferences
        session.speech_preferences = preferences
        session.updated_at = _utc_now_iso()
        if changed:
            self._bump_speech_preference_epoch(session.id)
        return session, True

    def session_workspace_id(self, session_id: str) -> str:
        """Return the workspace id a native Console session is bound to.

        Used by ``ConsoleAgentBridge.run_reply`` (task-6, settings-
        workspaces-folder-roots spec §3) to thread the RUNNING session's
        own workspace into ``BuiltinToolProvider`` -- never whatever
        workspace happens to be active in the UI by the time a tool
        actually fires.
        """
        return self._session_or_raise(session_id).workspace_id

    def session_is_ephemeral(self, session_id: str) -> bool:
        """Return whether a native Console session is temporary.

        Mirrors ``session_workspace_id`` exactly (final-review F4): used by
        ``ConsoleAgentBridge.run_reply`` to thread THIS run's own session
        into ``BuiltinToolProvider`` so it can refuse the write-shaped
        built-in tools (``create_note``/``update_note``/``write_file``) for
        a temporary chat -- never whatever session happens to be active in
        the UI by the time a tool actually fires.
        """
        return self._session_or_raise(session_id).ephemeral

    def replace_session_settings(
        self,
        session_id: str,
        settings: ConsoleSessionSettings,
        *,
        mark_user_work: bool = True,
        canonical_settings_baseline: ConsoleSessionSettings | None = None,
    ) -> ConsoleChatSession:
        """Replace in-memory settings for a native Console session."""
        session = self._session_or_raise(session_id)
        if mark_user_work and session.settings != settings:
            session.has_user_work = True
        elif not mark_user_work:
            if canonical_settings_baseline != settings:
                raise ValueError(
                    "Automatic settings replacement requires an exact canonical baseline."
                )
            session.canonical_settings_baseline = canonical_settings_baseline
        session.settings = settings
        self._bump_payload_revision(session_id)
        return session

    def session_draft(self, session_id: str) -> str:
        """Return the in-memory composer draft for a native Console session."""
        return self._session_or_raise(session_id).draft

    def set_session_draft(self, session_id: str, draft: str) -> ConsoleChatSession:
        """Replace the in-memory composer draft for a native Console session."""
        session = self._session_or_raise(session_id)
        session.draft = draft
        if draft:
            session.has_user_work = True
        return session

    def session_one_shot_prefill(self, session_id: str) -> str | None:
        """Return the armed one-shot response prefill for a session, if any."""
        return self._session_or_raise(session_id).one_shot_prefill

    def capture_policy_state(self, session_id: str) -> CapturePolicyState:
        """Return one immutable session policy view and the shared revision."""
        with self._capture_policy_lock:
            session = self._session_or_raise(session_id)
            return CapturePolicyState(
                next_detail=session.next_capture_detail,
                conversation_detail=session.capture_detail_override,
                next_revision=session.next_capture_detail_revision,
                policy_revision=self._capture_policy_revision,
                capture_revision=session.capture_revision,
                save_pending=session.capture_policy_save_pending,
            )

    def capture_revision(self, session_id: str) -> int:
        """Return the current Full-capture invalidation revision."""
        return self._session_or_raise(session_id).capture_revision

    def begin_capture_quiescence(self, session_id: str) -> bool:
        """Block exchange attachment/flush for one live session."""
        with self._capture_quiescence_lock:
            self._session_or_raise(session_id)
            if session_id in self._capture_quiescent_sessions:
                return False
            self._capture_quiescent_sessions.add(session_id)
            return True

    def capture_quiescent(self, session_id: str) -> bool:
        """Return whether one live session currently rejects capture writers."""
        with self._capture_quiescence_lock:
            return session_id in self._capture_quiescent_sessions

    def end_capture_quiescence(self, session_id: str) -> None:
        """Release one session's exchange writer fence."""
        with self._capture_quiescence_lock:
            self._capture_quiescent_sessions.discard(session_id)

    def stage_full_capture_purge(self, session_id: str) -> StagedCapturePurge:
        """Build every replacement needed after a durable Full-row delete."""
        session = self._session_or_raise(session_id)
        conversation_id = session.persisted_conversation_id
        durable_keys: frozenset[tuple[str, str, int]] = frozenset()
        if not session.ephemeral and conversation_id is not None:
            reader = getattr(
                self.persistence, "list_full_exchange_keys_for_conversation", None
            )
            if not callable(reader):
                raise RuntimeError("Full capture inventory is unavailable.")
            durable_keys = frozenset(reader(conversation_id))

        seen: set[int] = set()
        messages: list[ConsoleChatMessage] = []
        for message in self._nodes_by_session.get(session_id, {}).values():
            if id(message) not in seen:
                seen.add(id(message))
                messages.append(message)
        for _owner, marker in self._tool_markers_by_session.get(session_id, ()):
            if id(marker) not in seen:
                seen.add(id(marker))
                messages.append(marker)

        message_swaps: list[
            tuple[ConsoleChatMessage, tuple["ExchangeCapture", ...]]
        ] = []
        live_full_keys: set[tuple[str, str, int]] = set()
        remaining_run_tags: dict[str, set[str]] = {}
        remaining_capture_keys: dict[str, set[tuple[str, int, str]]] = {}
        for message in messages:
            exchanges = tuple(
                capture
                for capture in message.exchanges
                if capture.capture_detail is not CaptureDetail.FULL
            )
            persisted_id = message.persisted_message_id or message.id
            live_full_keys.update(
                (persisted_id, capture.run_tag, capture.seq)
                for capture in message.exchanges
                if capture.capture_detail is CaptureDetail.FULL
            )
            if exchanges != message.exchanges:
                message_swaps.append((message, exchanges))
            remaining_run_tags[message.id] = {capture.run_tag for capture in exchanges}
            remaining_capture_keys[message.id] = {
                (capture.run_tag, capture.seq, capture.status)
                for capture in exchanges
            }

        blob_cache = tuple(
            (
                message.id,
                MappingProxyType(
                    {
                        key: blob
                        for key, blob in self._exchange_blob_cache[message.id].items()
                        if key in remaining_capture_keys[message.id]
                    }
                ),
            )
            for message in messages
            if message.id in self._exchange_blob_cache
        )
        abandoned_tags = tuple(
            (
                message.id,
                frozenset(
                    self._abandoned_exchange_run_tags[message.id]
                    & remaining_run_tags[message.id]
                ),
            )
            for message in messages
            if message.id in self._abandoned_exchange_run_tags
        )
        capture_revisions = ((session, session.capture_revision + 1),)
        return StagedCapturePurge(
            session_id=session_id,
            conversation_id=conversation_id,
            expected_revision=session.capture_revision,
            durable_keys=durable_keys,
            message_swaps=tuple(message_swaps),
            blob_cache=blob_cache,
            abandoned_tags=abandoned_tags,
            capture_revisions=capture_revisions,
            removed_count=len(durable_keys | live_full_keys),
        )

    def commit_full_capture_purge(self, stage: StagedCapturePurge) -> int:
        """Delete durable Full rows, then publish only staged assignments."""
        session = self._session_or_raise(stage.session_id)
        if session.capture_revision != stage.expected_revision:
            raise CapturePurgeStaleError()
        if not session.ephemeral and stage.conversation_id is not None:
            deleter = getattr(
                self.persistence, "delete_full_exchanges_for_conversation", None
            )
            if not callable(deleter):
                raise RuntimeError("Full capture deletion is unavailable.")
            deleter(
                stage.conversation_id,
                expected_count=len(stage.durable_keys),
            )
        for message_id, cache in stage.blob_cache:
            self._exchange_blob_cache[message_id] = cache
        for message_id, tags in stage.abandoned_tags:
            self._abandoned_exchange_run_tags[message_id] = tags
        for message, exchanges in stage.message_swaps:
            message.exchanges = exchanges
        for target_session, revision in stage.capture_revisions:
            target_session.capture_revision = revision
        return stage.removed_count

    def hydrate_session_capture_policy(self, session_id: str) -> CapturePolicyReadResult:
        """Hydrate a persisted conversation override into process-local state."""
        with self._capture_policy_lock:
            session = self._session_or_raise(session_id)
            repository = self.capture_policy_repository
            if session.persisted_conversation_id is None:
                return CapturePolicyReadResult(CapturePolicyReadStatus.ABSENT, None)
            if repository is None:
                result = CapturePolicyReadResult(
                    CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT,
                    None,
                )
            else:
                try:
                    result = repository.read(session.persisted_conversation_id)
                except Exception:
                    result = CapturePolicyReadResult(
                        CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT,
                        None,
                    )
            if result.status is CapturePolicyReadStatus.FOUND:
                if result.policy is None:
                    result = CapturePolicyReadResult(
                        CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT,
                        None,
                    )
                else:
                    session.capture_detail_override = result.policy.detail
                    session.capture_policy_save_pending = False
            elif result.status is CapturePolicyReadStatus.ABSENT:
                session.capture_detail_override = None
                session.capture_policy_save_pending = False
            else:
                session.capture_detail_override = CaptureDetail.SAFE
                session.capture_policy_save_pending = True
            return result

    def _flush_staged_capture_policy(self, session: ConsoleChatSession) -> None:
        """Best-effort flush of an ephemeral policy after identity publication."""
        if session.capture_detail_override is None:
            return
        repository = self.capture_policy_repository
        if repository is None or session.persisted_conversation_id is None:
            session.capture_policy_save_pending = True
            return
        result = repository.replace(
            session.persisted_conversation_id,
            session.capture_detail_override,
        )
        session.capture_policy_save_pending = result.status not in {
            CapturePolicyWriteStatus.STORED,
            CapturePolicyWriteStatus.UNCHANGED,
        }

    def set_session_next_capture_detail(
        self,
        session_id: str,
        detail: CaptureDetail | None,
        *,
        expected_policy_revision: int,
    ) -> tuple[CaptureDetail | None, int, int]:
        """Arm or disarm a next-send detail behind the shared revision fence."""
        if detail is not None and not isinstance(detail, CaptureDetail):
            raise TypeError("detail must be CaptureDetail or None")
        with self._capture_policy_lock:
            session = self._session_or_raise(session_id)
            if (
                self._capture_policy_mutation is not None
                or self._capture_policy_revision != expected_policy_revision
            ):
                raise CapturePolicyStaleError
            session.next_capture_detail = detail
            session.next_capture_detail_revision += 1
            self._capture_policy_revision += 1
            return (
                session.next_capture_detail,
                session.next_capture_detail_revision,
                self._capture_policy_revision,
            )

    def consume_session_next_capture_detail(
        self,
        session_id: str,
        *,
        expected_next_revision: int,
    ) -> bool:
        """Clear only the exact next-send slot captured by admission."""
        with self._capture_policy_lock:
            session = self._session_or_raise(session_id)
            if session.next_capture_detail_revision != expected_next_revision:
                return False
            session.next_capture_detail = None
            session.next_capture_detail_revision += 1
            self._capture_policy_revision += 1
            return True

    def replace_session_capture_override(
        self,
        session_id: str,
        detail: CaptureDetail | None,
        *,
        expected_policy_revision: int,
        save_pending: bool = False,
    ) -> int:
        """Replace future conversation detail behind the shared revision fence."""
        if detail is not None and not isinstance(detail, CaptureDetail):
            raise TypeError("detail must be CaptureDetail or None")
        with self._capture_policy_lock:
            session = self._session_or_raise(session_id)
            if (
                self._capture_policy_mutation is not None
                or self._capture_policy_revision != expected_policy_revision
            ):
                raise CapturePolicyStaleError
            session.capture_detail_override = detail
            session.capture_policy_save_pending = bool(save_pending)
            self._capture_policy_revision += 1
            return self._capture_policy_revision

    def disarm_all_next_capture_details(self) -> int:
        """Disarm every live one-shot after the global kill switch turns off."""
        with self._capture_policy_lock:
            if self._capture_policy_mutation is not None:
                raise CapturePolicyStaleError
            changed = False
            for session in self._sessions.values():
                if session.next_capture_detail is None:
                    continue
                session.next_capture_detail = None
                session.next_capture_detail_revision += 1
                changed = True
            if changed:
                self._capture_policy_revision += 1
            return self._capture_policy_revision

    def advance_capture_policy_revision(
        self,
        *,
        expected_policy_revision: int,
        disarm_next: bool = False,
    ) -> int:
        """Advance the shared fence for a non-session policy mutation."""
        with self._capture_policy_lock:
            if (
                self._capture_policy_mutation is not None
                or self._capture_policy_revision != expected_policy_revision
            ):
                raise CapturePolicyStaleError
            if disarm_next:
                for session in self._sessions.values():
                    if session.next_capture_detail is None:
                        continue
                    session.next_capture_detail = None
                    session.next_capture_detail_revision += 1
            self._capture_policy_revision += 1
            return self._capture_policy_revision

    def reserve_capture_policy_mutation(
        self, *, expected_policy_revision: int
    ) -> object:
        """Reserve the shared policy owner before an external durable write."""
        with self._capture_policy_lock:
            if (
                self._capture_policy_mutation is not None
                or self._capture_policy_revision != expected_policy_revision
            ):
                raise CapturePolicyStaleError
            token = object()
            self._capture_policy_mutation = token
            self._capture_policy_revision += 1
            return token

    def publish_reserved_capture_safe(
        self,
        token: object,
        *,
        session_id: str,
        save_pending: bool,
    ) -> int:
        """Publish an explicit Safe override while retaining mutation ownership."""
        with self._capture_policy_lock:
            if self._capture_policy_mutation is not token:
                raise CapturePolicyStaleError
            session = self._session_or_raise(session_id)
            session.capture_detail_override = CaptureDetail.SAFE
            session.capture_policy_save_pending = bool(save_pending)
            return self._capture_policy_revision

    def finish_capture_policy_mutation(
        self,
        token: object,
        *,
        session_id: str | None = None,
        detail: CaptureDetail | None = None,
        save_pending: bool = False,
        disarm_next: bool = False,
    ) -> int:
        """Publish a reserved durable mutation and release its owner."""
        with self._capture_policy_lock:
            if self._capture_policy_mutation is not token:
                raise CapturePolicyStaleError
            try:
                if session_id is not None:
                    session = self._session_or_raise(session_id)
                    session.capture_detail_override = detail
                    session.capture_policy_save_pending = bool(save_pending)
                if disarm_next:
                    for session in self._sessions.values():
                        if session.next_capture_detail is not None:
                            session.next_capture_detail = None
                            session.next_capture_detail_revision += 1
                return self._capture_policy_revision
            finally:
                self._capture_policy_mutation = None

    def abandon_capture_policy_mutation(self, token: object) -> int:
        """Release a failed reserved mutation without publishing policy state."""
        with self._capture_policy_lock:
            if self._capture_policy_mutation is not token:
                raise CapturePolicyStaleError
            self._capture_policy_mutation = None
            return self._capture_policy_revision

    def set_session_one_shot_prefill(
        self, session_id: str, prefill: str | None
    ) -> ConsoleChatSession:
        """Arm (or clear, with ``None``) the one-shot response prefill."""
        session = self._session_or_raise(session_id)
        session.one_shot_prefill = prefill
        session.one_shot_prefill_revision += 1
        return session

    def session_one_shot_prefill_snapshot(
        self, session_id: str
    ) -> tuple[str | None, int]:
        """Return the current one-shot value and its opaque live revision."""

        session = self._session_or_raise(session_id)
        return session.one_shot_prefill, session.one_shot_prefill_revision

    def consume_session_one_shot_prefill(
        self, session_id: str, expected_revision: int
    ) -> bool:
        """Clear only the exact revision captured by an accepted turn."""

        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        session = self._session_or_raise(session_id)
        if session.one_shot_prefill_revision != expected_revision:
            return False
        session.one_shot_prefill = None
        session.one_shot_prefill_revision += 1
        return True

    def pending_attachments(self, session_id: str) -> list[PendingAttachment]:
        """Return the staged attachments for a session (stage order).

        Args:
            session_id: Native Console session ID.

        Returns:
            A copy of the staged attachments list, in stage order.

        Raises:
            KeyError: If the session is unknown.
        """
        return list(self._session_or_raise(session_id).pending_attachments)

    def add_pending_attachment(
        self, session_id: str, attachment: PendingAttachment
    ) -> bool:
        """Append a staged attachment; False (no-op) when at the cap.

        Args:
            session_id: Native Console session ID.
            attachment: Processed attachment to stage.

        Returns:
            True when staged; False when MAX_PENDING_ATTACHMENTS reached.

        Raises:
            KeyError: If the session is unknown.
        """
        session = self._session_or_raise(session_id)
        if len(session.pending_attachments) >= MAX_PENDING_ATTACHMENTS:
            return False
        session.pending_attachments.append(attachment)
        return True

    def clear_pending_attachments(self, session_id: str) -> ConsoleChatSession:
        """Remove all staged attachments from a session.

        Args:
            session_id: Native Console session ID.

        Returns:
            The updated session.

        Raises:
            KeyError: If the session is unknown.
        """
        session = self._session_or_raise(session_id)
        session.pending_attachments.clear()
        return session

    def consume_pending_attachment(self, session_id: str, attachment_id: str) -> bool:
        """Remove only the currently staged attachment with the exact identity.

        Args:
            session_id: Native Console session ID.
            attachment_id: Opaque process-local attachment identity.

        Returns:
            True when the exact attachment was removed; False when the
            identity is no longer staged.

        Raises:
            KeyError: If the session is unknown.
        """
        pending = self._session_or_raise(session_id).pending_attachments
        for index, attachment in enumerate(pending):
            if attachment.attachment_id == attachment_id:
                del pending[index]
                return True
        return False

    def pending_attachment(self, session_id: str) -> PendingAttachment | None:
        """Return the first staged attachment (legacy single accessor).

        Args:
            session_id: Native Console session ID.

        Returns:
            The first staged attachment, or None when nothing is staged.

        Raises:
            KeyError: If the session is unknown.
        """
        pending = self._session_or_raise(session_id).pending_attachments
        return pending[0] if pending else None

    def set_pending_attachment(
        self,
        session_id: str,
        attachment: PendingAttachment,
    ) -> ConsoleChatSession:
        """Replace all staged attachments with one (legacy semantics).

        Args:
            session_id: Native Console session ID.
            attachment: Processed attachment to stage for the next send.

        Returns:
            The updated session.

        Raises:
            KeyError: If the session is unknown.
        """
        session = self._session_or_raise(session_id)
        session.pending_attachments[:] = [attachment]
        return session

    def clear_pending_attachment(self, session_id: str) -> ConsoleChatSession:
        """Alias of clear_pending_attachments (legacy name).

        Args:
            session_id: Native Console session ID.

        Returns:
            The updated session.

        Raises:
            KeyError: If the session is unknown.
        """
        return self.clear_pending_attachments(session_id)

    def set_workspace_context(self, workspace_context: ConsoleWorkspaceContext) -> None:
        """Replace the active workspace context."""
        self.workspace_context = workspace_context

    def set_session_project_instruction_state(
        self,
        session_id: str,
        state: ProjectInstructionControlState,
    ) -> ConsoleChatSession:
        """Apply project-instruction controls and best-effort persist them.

        The in-memory state changes first and remains authoritative when the
        optional local-only write fails. Temporary sessions never write.

        Args:
            session_id: Native Console session identifier.
            state: Complete validated control state.

        Returns:
            The updated live session.
        """
        session = self._session_or_raise(session_id)
        session.project_instruction_state = state
        self._persist_project_instruction_state(session)
        return session

    def _persist_project_instruction_state(self, session: ConsoleChatSession) -> None:
        """Best-effort write one durable session's local control state."""
        conversation_id = session.persisted_conversation_id
        if (
            session.ephemeral
            or conversation_id is None
            or session.id in self._deferred_project_instruction_state_session_ids
        ):
            return
        setter = getattr(
            self.persistence, "set_conversation_console_project_context", None
        )
        if callable(setter):
            try:
                setter(
                    conversation_id=conversation_id,
                    project_context_json=encode_project_context_json(
                        session.project_instruction_state
                    ),
                )
            except Exception:
                pass
            else:
                return
        logger.warning(
            "project_instruction_state_write_failed: the updated choice "
            "may not survive restart."
        )

    def restore_state(
        self,
        *,
        sessions: Iterable[ConsoleChatSession],
        messages_by_session: Mapping[str, Iterable[ConsoleChatMessage]] | None = None,
        active_session_id: str | None = None,
    ) -> None:
        """Replace in-memory Console state with previously restored sessions.

        Args:
            sessions: Native Console sessions to load in display order.
            messages_by_session: Transcript messages keyed by session ID.
            active_session_id: Preferred active session after restoration.
        """
        restored_sessions = list(sessions)
        restored_messages = {
            session_id: tuple(messages)
            for session_id, messages in (messages_by_session or {}).items()
        }
        preserved_ephemeral = {
            session_id: recovery
            for session_id, recovery in self._dispatch_recoveries_by_session.items()
            if recovery.kind
            in {
                ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
                ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
            }
        }
        preserved_baselines = {
            session_id: baseline
            for session_id, baseline in self._dispatch_recovery_message_baselines.items()
            if session_id in preserved_ephemeral
        }
        sessions_by_id = {session.id: session for session in restored_sessions}
        for session_id, recovery in preserved_ephemeral.items():
            session = sessions_by_id.get(session_id)
            checkpoint = recovery.checkpoint
            message_roles = {
                message.id: message.role
                for message in restored_messages.get(session_id, ())
            }
            if (
                session is None
                or not session.ephemeral
                or checkpoint is None
                or message_roles.get(checkpoint.user_message_id)
                is not ConsoleMessageRole.USER
                or message_roles.get(recovery.assistant_message_id)
                is not ConsoleMessageRole.ASSISTANT
            ):
                raise RuntimeError(
                    "Unresolved temporary dispatch recovery cannot be replaced."
                )
        self._activate_session(None)
        if self.library_policy_coordinator is not None:
            for replaced_session_id in tuple(self._sessions):
                self.library_policy_coordinator.unregister_holder(replaced_session_id)
        self._sessions.clear()
        self._messages_by_session.clear()
        self._message_session_index.clear()
        self._pending_persistence_message_ids.clear()
        self._terminal_citation_finalizers.clear()
        self._provisional_terminal_selection_ids.clear()
        self._terminal_persistence_deferred_ids.clear()
        self._stream_chunks_by_message.clear()
        self._stream_materialized_counts.clear()
        self._character_emote_captures.clear()
        self._character_emote_feed_by_session.clear()
        self._sync_v2_message_versions.clear()
        self._roleplay_system_projection_candidates.clear()
        self._roleplay_message_projection_candidates.clear()
        # Pre-existing bug fixed while here: the regenerate base snapshots were
        # never cleared on restore, leaking across a state replacement.
        self._variant_stream_bases.clear()
        self._variant_restored_message_ids.clear()
        self._failed_retry_message_ids.clear()
        self._message_speech_revisions.clear()
        self._message_completion_generations.clear()
        # M2: both keyed by message id, same as the caches immediately
        # above -- previously left uncleared here, so a restore (session
        # switch / app restart replay, distinct from delete_message and
        # session-close, the only two call sites that used to drop entries)
        # could leave stale entries keyed by message ids that no longer
        # exist in the replaced state.
        self._abandoned_exchange_run_tags.clear()
        self._exchange_blob_cache.clear()
        self._payload_revisions.clear()
        self._conversation_context_epochs.clear()
        self._speech_preference_epochs.clear()
        self._nodes_by_session.clear()
        self._children_by_parent.clear()
        self._native_parent_by_message.clear()
        self._active_leaf_by_session.clear()
        self._context_summary_by_session.clear()
        self._pending_workspace_projections.clear()
        self._dispatch_recoveries_by_session.clear()
        self._dispatch_recovery_message_baselines.clear()
        self._dispatch_recovery_queue_hydration_pending.clear()

        for session in restored_sessions:
            restored_holder = ConsoleLibraryPolicyHolder(
                snapshot=session.library_policy_holder.snapshot,
                explicitly_staged=session.library_policy_holder.explicitly_staged,
                save_pending=session.library_policy_holder.save_pending,
            )
            restored_session = replace(
                session,
                library_policy_holder=restored_holder,
            )
            self._sessions[session.id] = restored_session
            if self.library_policy_coordinator is not None:
                self.library_policy_coordinator.register_holder(
                    session.id,
                    restored_session.persisted_conversation_id,
                    restored_holder,
                )
            if (
                restored_session.persisted_conversation_id is not None
                and restored_session.workspace_id != CONSOLE_GLOBAL_WORKSPACE_ID
            ):
                self._pending_workspace_projections[session.id] = (
                    restored_session.persisted_conversation_id
                )
            self._bump_speech_preference_epoch(session.id)
            self._nodes_by_session[session.id] = {}
            self._children_by_parent[session.id] = {}
            self._active_leaf_by_session[session.id] = None
            self._context_summary_by_session[session.id] = (None, None)
            self._conversation_context_epochs[session.id] = 0
            self._messages_by_session[session.id] = []
            self._ingest_linear_messages(
                session.id, restored_messages.get(session.id, ())
            )
            self._bump_payload_revision(session.id)

        self._dispatch_recoveries_by_session.update(preserved_ephemeral)
        self._dispatch_recovery_message_baselines.update(preserved_baselines)

        if active_session_id in self._sessions:
            self._activate_session(active_session_id)
        elif self._sessions:
            self._activate_session(next(iter(self._sessions)))

    def end_app_runtime(self) -> None:
        """Drop every volatile recovery projection at explicit app teardown."""

        with self._preparation_lock:
            self._dispatch_recoveries_by_session.clear()
            self._dispatch_recovery_message_baselines.clear()
            self._dispatch_recovery_queue_hydration_pending.clear()

    @staticmethod
    def _set_message_attachments(
        message: ConsoleChatMessage,
        attachments: Sequence[MessageAttachment],
    ) -> None:
        """Set a message's attachments tuple and mirror #0 into the scalars.

        Every attachments mutation MUST flow through here — the scalar
        image fields are a read-compatibility mirror of attachments[0].
        Positions are re-based sequentially from 0 in the given order.
        """
        rebased = tuple(
            replace(attachment, position=index)
            for index, attachment in enumerate(attachments)
        )
        message.attachments = rebased
        first = rebased[0] if rebased else None
        message.image_data = first.data if first else None
        message.image_mime_type = first.mime_type if first else None
        message.attachment_label = (
            first.display_name if first and first.display_name else None
        )

    def append_message(
        self,
        session_id: str,
        *,
        role: ConsoleMessageRole,
        content: str,
        persist: bool = False,
        attachments: Sequence[MessageAttachment] = (),
        image_data: bytes | None = None,
        image_mime_type: str | None = None,
        attachment_label: str | None = None,
        terminal_citation_finalizer: TerminalCitationFinalizer | None = None,
        defer_terminal_persistence: bool = False,
        tool_output_full: str | None = None,
        tool_diff: tuple[str, str, str] | None = None,
        change_review_run_id: str | None = None,
        metadata: "MessageMetadata | None" = None,
        activity_presentation: ConsoleActivityPresentation | None = None,
        message_id: str | None = None,
    ) -> ConsoleChatMessage:
        """Append a message; scalar image kwargs become a one-item tuple.

        ``metadata`` (task-2364) records structured facts about the turn
        (engine provenance, interrupted, transcript status) at creation
        time, so a row that knows its own provenance writes it with the
        create instead of chasing it with a second update.

        ``tool_diff`` (TASK-1366) is the raw (path, before, after) capture
        behind a file-writing TOOL marker -- session-only, never persisted
        (see ``ConsoleChatMessage.tool_diff``).

        ``activity_presentation`` is session-only structured display state;
        it is attached only to the in-memory message and never serialized.
        """
        self._session_or_raise(session_id)
        effective = tuple(attachments)
        if not effective and image_data is not None:
            effective = (
                MessageAttachment(
                    data=image_data,
                    mime_type=image_mime_type or "image/png",
                    display_name=attachment_label or "",
                    position=0,
                ),
            )
        if terminal_citation_finalizer is not None:
            if not callable(terminal_citation_finalizer):
                raise ValueError("terminal_citation_finalizer must be callable")
            if type(content) is not str:
                raise ValueError(
                    "terminal_citation_finalizer requires exact string content"
                )
            if role is not ConsoleMessageRole.ASSISTANT or content != "" or effective:
                raise ValueError(
                    "terminal_citation_finalizer requires an empty, "
                    "attachment-free assistant placeholder"
                )
        if defer_terminal_persistence:
            if type(content) is not str:
                raise ValueError(
                    "defer_terminal_persistence requires exact string content"
                )
            if role is not ConsoleMessageRole.ASSISTANT or content != "" or effective:
                raise ValueError(
                    "defer_terminal_persistence requires an empty, "
                    "attachment-free assistant placeholder"
                )
        arm_finalizer = (
            terminal_citation_finalizer is not None
            and persist
            and self._citation_persistence_ready()
        )
        arm_provisional_selection = defer_terminal_persistence
        arm_terminal_deferral = (
            persist
            and self.persistence is not None
            and (defer_terminal_persistence or arm_finalizer)
        )
        message = ConsoleChatMessage(
            id=message_id or str(uuid4()),
            role=role,
            content=content,
            status=self._initial_status(role=role, content=content),
            tool_output_full=tool_output_full,
            tool_diff=tool_diff,
            change_review_run_id=change_review_run_id,
            metadata=metadata,
            activity_presentation=activity_presentation,
        )
        self._set_message_attachments(message, effective)
        if attachment_label and effective and not effective[0].display_name:
            message.attachment_label = attachment_label
        self._sessions[session_id].updated_at = _utc_now_iso()
        if role is ConsoleMessageRole.TOOL:
            # Display-only agent marker (TOOL-marker invariant): register the
            # session index and append to the active-path view for display, but
            # NEVER become a tree node, the active leaf, or a parent -- otherwise
            # the next real message would parent at a marker and corrupt the
            # chain even in linear agent chats. Returns without persisting.
            #
            # TASK-1842: the view alone is not enough. `_recompute_active_path`
            # is the SINGLE writer of `_messages_by_session` and rebuilds it
            # from tree nodes only, so a marker appended here was erased by the
            # very next ordinary message -- the user watched an agent's tool
            # trace vanish. Anchor it to the node it followed so the rebuild can
            # splice it back at the right place, WITHOUT it ever becoming a node.
            self._message_session_index[message.id] = session_id
            anchor = self._active_leaf_by_session.get(session_id)
            self._tool_markers_by_session.setdefault(session_id, []).append(
                (anchor, message)
            )
            self._messages_by_session[session_id].append(message)
            # Trajectory sidecar (schema v38): the marker itself is never
            # persisted, but its tool_call/tool_result records ARE -- keyed
            # to the anchor (parent assistant) message. Best-effort.
            anchor_node = self._nodes_by_session.get(session_id, {}).get(anchor)
            self._record_trajectory_tool_marker(
                session_id, anchor_node, content, tool_output_full
            )
            return self._snapshot(message)
        old_leaf = self._active_leaf_by_session[session_id]
        self._register_tree_node(session_id, message, parent_native_id=old_leaf)
        self._active_leaf_by_session[session_id] = message.id
        self._recompute_active_path(session_id)
        if arm_finalizer:
            assert terminal_citation_finalizer is not None
            self._terminal_citation_finalizers[message.id] = terminal_citation_finalizer
        if arm_provisional_selection:
            self._provisional_terminal_selection_ids.add(message.id)
        if arm_terminal_deferral:
            self._terminal_persistence_deferred_ids.add(message.id)
        try:
            if persist:
                self._persist_new_message_or_defer(
                    session_id=session_id, message=message
                )
        except Exception:
            if arm_finalizer or arm_provisional_selection or arm_terminal_deferral:
                self.clear_terminal_citation_state(message.id)
            raise
        self._bump_payload_revision(session_id)
        return self._snapshot(message)

    def create_sibling(
        self,
        anchor_message_id: str,
        *,
        role: ConsoleMessageRole,
        content: str = "",
        persist: bool = False,
        attachments: Sequence[MessageAttachment] = (),
    ) -> ConsoleChatMessage:
        """Fork a new node alongside ``anchor_message_id`` and make it active.

        This is the primitive regenerate uses: unlike ``append_message`` --
        which always parents the new node at the CURRENT active leaf -- the
        new node here is parented at the anchor's OWN native parent (a
        SIBLING of the anchor, not a child of it). Registering it via
        ``_register_tree_node`` adds it to the anchor's parent's ordered
        child list beside the anchor (so ``siblings_at`` reports both), then
        the session's active leaf is retargeted at the new node and the
        active-path view is recomputed (Task 3's single writer).

        When the anchor is mid-conversation (has descendants of its own),
        that old tail drops off the now-recomputed active path -- it is not
        deleted, just no longer on the visible branch, and remains reachable
        by swiping back (``set_active_leaf`` to any node in the old branch).

        Args:
            anchor_message_id: Native id of the node to fork alongside
                (typically the assistant message being regenerated).
            role: Role for the new sibling message.
            attachments: Attachments to set on the new sibling (task-573:
                Edit & resend carries the anchor's attachments onto the
                fork). Same seam ``append_message`` uses; empty by default.
            content: Initial content. An empty-content assistant sibling
                starts ``"pending"`` (mirrors ``append_message``), ready to
                receive stream chunks via ``append_stream_chunk``.
            persist: When True, write the new node through to durable
                storage immediately, using the same persist path
                ``append_message(persist=True)`` uses. Ordering is
                deliberate: the active leaf is retargeted and the
                active-path view recomputed BEFORE this write (so the Sync v2
                sequence helper, which walks the active-path view, sees the
                new node on-path and emits its real on-path ordinal instead
                of ``None``), and the DB active-leaf pointer write-through
                (``_persist_active_leaf``) runs AFTER it (so, when the
                session already owns a persisted conversation, it observes
                the new node's freshly assigned ``persisted_message_id``
                instead of the pre-persist ``None``).

        Returns:
            A snapshot of the newly created sibling node.

        Raises:
            KeyError: If ``anchor_message_id`` is not a known tree node.
        """
        self._message_or_raise(anchor_message_id)
        session_id = self._message_session_index[anchor_message_id]
        parent_native_id = self._native_parent_by_message.get(anchor_message_id)
        message = ConsoleChatMessage(
            role=role,
            content=content,
            status=self._initial_status(role=role, content=content),
        )
        # task-573: a fork can carry the anchor's attachments (Edit & resend
        # of an image-bearing user turn); same seam ``append_message`` uses.
        self._set_message_attachments(message, tuple(attachments))
        self._sessions[session_id].updated_at = _utc_now_iso()
        self._register_tree_node(session_id, message, parent_native_id=parent_native_id)
        # Retarget the active leaf and rematerialize the active-path view
        # BEFORE persisting so the Sync v2 sequence helper (which walks the
        # active-path VIEW) sees the new node on-path and emits its real
        # on-path ordinal, not ``None``. This intentionally does NOT route
        # through ``set_active_leaf``, whose bundled ordering also writes the
        # DB active-leaf pointer -- that pointer write must happen AFTER
        # persistence to capture the node's real ``persisted_message_id``.
        self._active_leaf_by_session[session_id] = message.id
        self._recompute_active_path(session_id)
        if persist:
            self._persist_new_message_or_defer(session_id=session_id, message=message)
        # Write-through the DB active-leaf pointer now that (for persist=True)
        # the node owns a persisted id. For the persist=False path this mirrors
        # the old ``set_active_leaf`` call with a still-``None`` id, which is fine.
        self._persist_active_leaf(session_id, message.id)
        self._bump_payload_revision(session_id)
        self._bump_conversation_context_epoch(session_id)
        return self._snapshot(message)

    def append_generation_message(
        self,
        session_id: str,
        *,
        content: str,
        variants: Sequence[tuple[bytes, str, GenerationVariantMeta]],
        persist: bool = False,
    ) -> ConsoleChatMessage:
        """Append an assistant image-generation message with N variants.

        Builds the full 0..N-1 attachment list and the index-aligned
        ``generation_metadata`` tuple from ``variants`` in one shot, then
        (optionally) persists both atomically via
        ``create_message(attachments=..., generation_metadata=...)`` --
        the ONE place the full attachment list is authoritative and safe to
        send, because these bytes are fresh (never rehydrated-without-bytes
        like a reloaded message's attachments can be).

        Args:
            session_id: Target Console session id.
            content: Short marker text for the message body (e.g.
                ``"[image] a red dragon"``).
            variants: Ordered ``(data, mime_type, meta)`` tuples; index i
                becomes attachment position i and
                ``generation_metadata[i]``.
            persist: When True, persist the message and its sidecar
                metadata through the durable adapter immediately.

        Returns:
            The LIVE internal message node -- deliberately NOT a snapshot,
            unlike most other append methods. Callers holding this
            reference observe in-place mutations from subsequent
            ``keep_generation_variant``/``append_generation_variant`` calls
            against this message's id, without needing to re-fetch.

        Raises:
            KeyError: If ``session_id`` is unknown.
        """
        self._session_or_raise(session_id)
        attachments = tuple(
            MessageAttachment(
                data=data, mime_type=mime_type, display_name="", position=index
            )
            for index, (data, mime_type, _meta) in enumerate(variants)
        )
        message = ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content=content,
            status=self._initial_status(
                role=ConsoleMessageRole.ASSISTANT, content=content
            ),
        )
        self._set_message_attachments(message, attachments)
        message.generation_metadata = tuple(meta for _, _, meta in variants)
        self._sessions[session_id].updated_at = _utc_now_iso()
        old_leaf = self._active_leaf_by_session[session_id]
        self._register_tree_node(session_id, message, parent_native_id=old_leaf)
        self._active_leaf_by_session[session_id] = message.id
        self._recompute_active_path(session_id)
        if persist:
            self._persist_new_message_or_defer(session_id=session_id, message=message)
        self._bump_payload_revision(session_id)
        return message

    def merge_persisted_generation_message(
        self, session_id: str, message_id: str
    ) -> ConsoleChatMessage | None:
        """Idempotently merge one exact persisted PNG generation message.

        This narrow remount-reconciliation seam reads through the persistence
        adapter's declared database boundary and existing attachment/metadata
        readers. It never rewrites persistence or mutates an existing message.

        Args:
            session_id: Native Console session receiving the restored message.
            message_id: Exact durable generation-message ID to read.

        Returns:
            A snapshot of the existing or newly merged message, or None when
            the durable row is missing or is not the expected one-PNG,
            one-metadata-row generation shape.

        Raises:
            KeyError: If the session is unknown.
        """
        session = self._session_or_raise(session_id)
        nodes = self._nodes_by_session[session_id]
        for existing in nodes.values():
            if existing.persisted_message_id == message_id:
                return self._snapshot(existing)
        if message_id in nodes or self.persistence is None:
            return None
        db = self.persistence.db
        if db is None:
            return None
        read_message = getattr(db, "get_message_by_id", None)
        read_attachments = getattr(
            self.persistence, "get_attachments_for_messages", None
        )
        read_metadata = getattr(
            self.persistence, "get_generation_metadata_for_messages", None
        )
        if not all(
            callable(reader)
            for reader in (read_message, read_attachments, read_metadata)
        ):
            return None

        row = read_message(message_id)
        if not isinstance(row, Mapping):
            return None
        if (
            row.get("id") != message_id
            or row.get("conversation_id") != session.persisted_conversation_id
            or row.get("sender") != ConsoleMessageRole.ASSISTANT.value
            or type(row.get("content")) is not str
            or type(row.get("image_data")) is not bytes
            or row.get("image_mime_type") != "image/png"
        ):
            return None
        extra_by_message = read_attachments([message_id])
        metadata_by_message = read_metadata([message_id])
        if not isinstance(extra_by_message, Mapping) or not isinstance(
            metadata_by_message, Mapping
        ):
            return None
        if extra_by_message.get(message_id, []) != []:
            return None
        metadata_rows = metadata_by_message.get(message_id)
        if (
            not isinstance(metadata_rows, list)
            or len(metadata_rows) != 1
            or not isinstance(metadata_rows[0], Mapping)
            or metadata_rows[0].get("position") != 0
        ):
            return None

        image_data = row["image_data"]
        metadata = GenerationVariantMeta.from_row(metadata_rows[0])
        message = ConsoleChatMessage(
            id=message_id,
            persisted_message_id=message_id,
            parent_message_id=row.get("parent_message_id"),
            role=ConsoleMessageRole.ASSISTANT,
            content=row["content"],
            status="complete",
            image_data=image_data,
            image_mime_type="image/png",
            attachments=(
                MessageAttachment(
                    data=image_data,
                    mime_type="image/png",
                    display_name="",
                    position=0,
                ),
            ),
            generation_metadata=(metadata,),
        )
        parent_native_id = next(
            (
                node.id
                for node in nodes.values()
                if node.persisted_message_id == message.parent_message_id
            ),
            None,
        )
        self._register_tree_node(session_id, message, parent_native_id=parent_native_id)
        self._active_leaf_by_session[session_id] = message.id
        self._recompute_active_path(session_id)
        self._bump_payload_revision(session_id)
        return self._snapshot(message)

    def append_video_message(
        self,
        session_id: str,
        *,
        video_metadata: "VideoGenerationMetadata",
        persist: bool = False,
        message_id: str | None = None,
    ) -> ConsoleChatMessage:
        """Append an assistant video-generation message (task-3401.4).

        The video's bytes are NOT on this message -- they live (ephemerally)
        in the VideoStore keyed by this message's id (ADR-044). The message
        carries the ``[video] <slug>`` marker as content and the structured
        ``video_metadata`` (persisted as a namespaced payload in the
        local-only ``metadata_json`` column) from which the card renders
        both the live video and, after restart/expiry, the named tombstone
        with a regenerate action. No attachments are created -- the v25
        generation-metadata sidecar's position-alignment invariant is
        untouched by design.

        Args:
            session_id: Target Console session id.
            video_metadata: Structured facts about the generated video;
                ``video_metadata.name`` is the slug the marker renders.
            persist: When True, persist through the durable adapter.
            message_id: Optional pre-allocated native id (task-3401.5): the
                caller saves the video bytes under this id BEFORE appending
                (the VideoStore keys by message id), so the id must be known
                ahead of the append. Defaults to a fresh uuid4.

        Returns:
            The LIVE internal message node (parity with
            ``append_generation_message``).

        Raises:
            KeyError: If ``session_id`` is unknown.
        """
        self._session_or_raise(session_id)
        content = video_content_marker(video_metadata.name)
        message = ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content=content,
            status=self._initial_status(
                role=ConsoleMessageRole.ASSISTANT, content=content
            ),
            video_metadata=video_metadata,
            **({"id": message_id} if message_id else {}),
        )
        self._sessions[session_id].updated_at = _utc_now_iso()
        old_leaf = self._active_leaf_by_session[session_id]
        self._register_tree_node(session_id, message, parent_native_id=old_leaf)
        self._active_leaf_by_session[session_id] = message.id
        self._recompute_active_path(session_id)
        if persist:
            self._persist_new_message_or_defer(session_id=session_id, message=message)
        self._bump_payload_revision(session_id)
        return message

    def append_generation_variant(
        self,
        session_id: str,
        message_id: str,
        *,
        data: bytes,
        mime_type: str,
        meta: GenerationVariantMeta,
        persist: bool = True,
    ) -> int:
        """Append one new generated variant to an existing generation message.

        Extends the in-memory attachments tuple (through
        ``_set_message_attachments``, the store's one mutation seam) and the
        ``generation_metadata`` tuple in lockstep, then -- when persisting --
        writes through via the probed, narrow
        ``persistence.append_message_attachment`` (a single INSERT; never
        the full-list ``update_message_content(attachments=...)`` rewrite,
        which would re-send and risk nulling every other variant's bytes).

        Args:
            session_id: Session the message belongs to (used to touch
                ``updated_at``; the message itself is resolved by id).
            message_id: Target message id; must already be a generation
                message (non-empty ``generation_metadata``).
            data: The new variant's image bytes.
            mime_type: The new variant's MIME type.
            meta: Generation metadata for the new variant.
            persist: When True, write through the probed narrow append op.
                Silently skipped (in-memory-only) when no persistence
                adapter is configured, the message was never persisted, or
                the adapter doesn't implement the op.

        Returns:
            The position assigned to the new variant -- index-aligned with
            the updated ``generation_metadata`` tuple.

        Raises:
            KeyError: If ``session_id`` or ``message_id`` is unknown.
        """
        self._session_or_raise(session_id)
        message = self._message_or_raise(message_id)
        owner_session_id = self._message_session_index[message.id]
        if not message.generation_metadata:
            raise ValueError(
                "append_generation_variant requires a generation message "
                "(non-empty generation_metadata)."
            )
        new_position = len(message.attachments)
        new_attachment = MessageAttachment(
            data=data, mime_type=mime_type, display_name="", position=new_position
        )
        self._set_message_attachments(message, (*message.attachments, new_attachment))
        message.generation_metadata = (*message.generation_metadata, meta)
        self._sessions[session_id].updated_at = _utc_now_iso()
        if (
            persist
            and self.persistence is not None
            and message.persisted_message_id is not None
            and getattr(self.persistence, "append_message_attachment", None) is not None
        ):
            persisted_position = self.persistence.append_message_attachment(
                message.persisted_message_id,
                data=data,
                mime_type=mime_type,
                display_name="",
                generation_metadata=meta.to_row(new_position),
            )
            if persisted_position is not None and persisted_position != new_position:
                raise RuntimeError(
                    f"generation variant position drift: store computed {new_position}, "
                    f"persistence assigned {persisted_position}"
                )
        self._bump_payload_revision(session_id)
        if self._message_is_on_active_path(message_id):
            self._bump_conversation_context_epoch(owner_session_id)
        return new_position

    def keep_generation_variant(
        self,
        session_id: str,
        message_id: str,
        *,
        position: int,
        persist: bool = True,
    ) -> None:
        """Promote a browsed variant to be the message's canonical (position-0) image.

        Reorders the in-memory attachments tuple -- the kept variant and
        position 0 SWAP places -- and the ``generation_metadata`` tuple in
        lockstep, through ``_set_message_attachments``. When the in-memory
        bytes for either affected variant are ``None`` (e.g. a
        rehydrated-without-bytes message), the swap still happens in memory
        (bytes stay ``None``); this is safe because persistence performs the
        real swap by reading bytes from the DB itself, via the narrow
        ``keep_message_attachment`` op -- NEVER the full-list
        ``update_message_content(attachments=...)`` rewrite, which would
        NULL any byte-less variant it re-sends (the spec's footgun
        scenario).

        Args:
            session_id: Session the message belongs to (used to touch
                ``updated_at``).
            message_id: Target message id.
            position: The attachment position (>= 1) to promote to
                canonical.
            persist: When True, write through the probed narrow keep op
                (see ``append_generation_variant`` for the skip conditions).

        Raises:
            KeyError: If ``session_id`` or ``message_id`` is unknown.
            ValueError: If ``position`` is out of range for this message's
                attachments.
        """
        self._session_or_raise(session_id)
        message = self._message_or_raise(message_id)
        owner_session_id = self._message_session_index[message.id]
        if position <= 0 or position >= len(message.attachments):
            raise ValueError(
                f"No attachment at position {position} to keep for message"
                f" {message_id}."
            )
        reordered = list(message.attachments)
        reordered[0], reordered[position] = reordered[position], reordered[0]
        self._set_message_attachments(message, tuple(reordered))
        if message.generation_metadata:
            reordered_metadata = list(message.generation_metadata)
            reordered_metadata[0], reordered_metadata[position] = (
                reordered_metadata[position],
                reordered_metadata[0],
            )
            message.generation_metadata = tuple(reordered_metadata)
        self._sessions[session_id].updated_at = _utc_now_iso()
        if (
            persist
            and self.persistence is not None
            and message.persisted_message_id is not None
            and getattr(self.persistence, "keep_message_attachment", None) is not None
        ):
            self.persistence.keep_message_attachment(
                message.persisted_message_id, position
            )
        self._bump_payload_revision(session_id)
        if self._message_is_on_active_path(message_id):
            self._bump_conversation_context_epoch(owner_session_id)

    def hydrate_generation_metadata(
        self,
        session_id: str,
        rows_by_message: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> None:
        """Populate messages' ``generation_metadata`` from DB sidecar rows.

        Called after a conversation's messages have been restored as tree
        nodes (so ``persisted_message_id`` is already set on each), using
        rows the caller batch-fetched via the probed
        ``persistence.get_generation_metadata_for_messages`` -- keyed by
        PERSISTED message id, matching that method's contract. Rows are
        converted with ``GenerationVariantMeta.from_row`` and assigned in
        the given (position-ordered) sequence; a message absent from
        ``rows_by_message``, or mapped to an empty sequence, is left alone
        (stays a non-generation message when its ``generation_metadata`` was
        already empty).

        Args:
            session_id: Session whose messages should be hydrated.
            rows_by_message: Mapping of PERSISTED message id to its
                position-ordered generation-metadata row sequence.
        """
        self._session_or_raise(session_id)
        nodes = self._nodes_by_session.get(session_id, {})
        for message in nodes.values():
            persisted_id = message.persisted_message_id
            if persisted_id is None:
                continue
            rows = rows_by_message.get(persisted_id)
            if not rows:
                continue
            message.generation_metadata = tuple(
                GenerationVariantMeta.from_row(row) for row in rows
            )

    def messages_for_session(self, session_id: str) -> list[ConsoleChatMessage]:
        """Return messages for a session in transcript order."""
        self._session_or_raise(session_id)
        for message in self._messages_by_session[session_id]:
            self._materialize_stream_buffer(message)
        return [
            self._snapshot(message) for message in self._messages_by_session[session_id]
        ]

    def read_only_messages_for_session(
        self, session_id: str
    ) -> list[ConsoleChatMessage]:
        """Return current transcript snapshots without mutating the store.

        Args:
            session_id: Session whose transcript should be projected.

        Returns:
            Detached snapshots including the latest buffered streaming text.

        Raises:
            KeyError: If ``session_id`` does not identify a stored session.
        """
        self._session_or_raise(session_id)
        snapshots: list[ConsoleChatMessage] = []
        for message in self._messages_by_session[session_id]:
            snapshot = self._snapshot(message)
            buffer = self._stream_chunks_by_message.get(message.id)
            if buffer and self._stream_materialized_counts.get(message.id) != len(
                buffer
            ):
                snapshot.content = "".join(buffer)
            snapshots.append(snapshot)
        return snapshots

    def get_message(self, message_id: str) -> ConsoleChatMessage:
        """Return a message by native message ID."""
        message = self._message_or_raise(message_id)
        self._materialize_stream_buffer(message)
        return self._snapshot(message)

    def issue_tts_message_speech_snapshot(
        self,
        message_id: str,
        *,
        presentation_context: ConsolePresentationContext | None = None,
    ) -> TTSMessageSpeechSnapshot:
        """Issue a trusted snapshot for one speakable active-path message.

        Args:
            message_id: Native Console message selected by the user.
            presentation_context: Optional live identity used to resolve trusted
                character-template content. ``None`` preserves neutral callers.

        Returns:
            An immutable snapshot bound to the exact selected text and
            current trusted session authorship.

        Raises:
            ConsoleSpeechSnapshotRejected: If the message is missing,
                inactive, incomplete, non-assistant, blank, or cannot be
                durably version-fenced.
        """
        session_id = self._message_session_index.get(message_id)
        if session_id is None:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MISSING_MESSAGE
            )
        if session_id != self.active_session_id:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.SESSION_CHANGED
            )
        session = self._sessions.get(session_id)
        message = self._nodes_by_session.get(session_id, {}).get(message_id)
        if session is None or message is None:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MISSING_MESSAGE
            )
        if (
            message.persisted_message_id is not None
            and session.persisted_conversation_id is None
        ):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        if message_id not in self.active_path_message_ids(session_id):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        raw_content, selected_variant_id = self._speech_selection(message)
        if presentation_context is not None:
            if not isinstance(presentation_context, ConsolePresentationContext):
                raise ValueError(
                    "presentation_context must be ConsolePresentationContext or None"
                )
            raw_content = resolve_console_message_presentation(
                self._snapshot(message), presentation_context
            ).content
        if (
            message.role is not ConsoleMessageRole.ASSISTANT
            or message.status != "complete"
            or not raw_content.strip()
        ):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_NOT_SPEAKABLE
            )
        speech_revision = self._message_speech_revisions.get(message_id)
        if type(speech_revision) is not int or speech_revision < 0:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        if (
            session.assistant_kind is not None
            and type(session.assistant_kind) is not str
        ):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.AUTHORSHIP_CHANGED
            )
        persisted_version = self._persisted_message_version_or_reject(message)
        return TTSMessageSpeechSnapshot(
            session_id=session_id,
            message_id=message.id,
            persisted_conversation_id=session.persisted_conversation_id,
            persisted_message_id=message.persisted_message_id,
            raw_content=raw_content,
            selected_variant_id=selected_variant_id,
            speech_revision=speech_revision,
            persisted_message_version=persisted_version,
            role=message.role,
            status=message.status,
            assistant_kind=session.assistant_kind,
            character_ref=session.character_ref(),
        )

    def validate_tts_message_speech_snapshot(
        self,
        snapshot: TTSMessageSpeechSnapshot,
        *,
        presentation_context: ConsolePresentationContext | None = None,
    ) -> str:
        """Revalidate an issued Console speech snapshot against live state.

        Args:
            snapshot: Immutable snapshot previously issued by this store.
            presentation_context: Optional fresh identity used to re-resolve
                trusted character-template content before comparison.

        Returns:
            The captured exact raw content after every identity, state,
            authorship, and durable-version check succeeds.

        Raises:
            ConsoleSpeechSnapshotRejected: If any captured fact is stale or
                cannot be re-established.
        """
        if type(snapshot) is not TTSMessageSpeechSnapshot:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        if snapshot.session_id != self.active_session_id:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.SESSION_CHANGED
            )
        session = self._sessions.get(snapshot.session_id)
        if session is None:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.SESSION_CHANGED
            )
        owner_session_id = self._message_session_index.get(snapshot.message_id)
        if owner_session_id is None:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MISSING_MESSAGE
            )
        if owner_session_id != snapshot.session_id:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        message = self._nodes_by_session.get(snapshot.session_id, {}).get(
            snapshot.message_id
        )
        if message is None:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MISSING_MESSAGE
            )
        if (
            message.persisted_message_id is not None
            and session.persisted_conversation_id is None
        ):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        if snapshot.message_id not in self.active_path_message_ids(snapshot.session_id):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        raw_content, selected_variant_id = self._speech_selection(message)
        if presentation_context is not None:
            if not isinstance(presentation_context, ConsolePresentationContext):
                raise ValueError(
                    "presentation_context must be ConsolePresentationContext or None"
                )
            raw_content = resolve_console_message_presentation(
                self._snapshot(message), presentation_context
            ).content
        if (
            message.role is not ConsoleMessageRole.ASSISTANT
            or message.status != "complete"
            or not raw_content.strip()
        ):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_NOT_SPEAKABLE
            )
        if (
            message.id != snapshot.message_id
            or message.role is not snapshot.role
            or message.status != snapshot.status
            or selected_variant_id != snapshot.selected_variant_id
            or raw_content != snapshot.raw_content
            or self._message_speech_revisions.get(message.id)
            != snapshot.speech_revision
            or session.persisted_conversation_id != snapshot.persisted_conversation_id
            or message.persisted_message_id != snapshot.persisted_message_id
        ):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        if (
            session.assistant_kind != snapshot.assistant_kind
            or session.character_ref() != snapshot.character_ref
        ):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.AUTHORSHIP_CHANGED
            )
        current_version = self._persisted_message_version_or_reject(message)
        if current_version != snapshot.persisted_message_version:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.PERSISTED_VERSION_CHANGED
            )
        return snapshot.raw_content

    def replace_deferred_terminal_body(
        self,
        message_id: str,
        selected_body: str,
    ) -> ConsoleChatMessage:
        """Atomically replace one provisional assistant body's stream state."""
        message = self._message_or_raise(message_id)
        if message.id not in self._provisional_terminal_selection_ids:
            raise ValueError("Message is not eligible for provisional selection.")
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages support provisional selection.")
        if message.attachments or message.image_data is not None:
            raise ValueError("Attached messages do not support provisional selection.")
        if message.status not in {"pending", "streaming"}:
            raise ValueError(
                f"Cannot replace a {message.status} provisional message body."
            )
        if type(selected_body) is not str or selected_body == "":
            raise ValueError("Selected body must be a non-empty string.")
        try:
            selected_body_size = len(str.encode(selected_body, "utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("Selected body must be valid UTF-8 text.") from exc
        if selected_body_size > ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX:
            raise ValueError(
                "Selected body exceeds the answer-attempt UTF-8 byte limit."
            )

        capture = self._character_emote_captures.get(message.id)
        replacement_events: tuple[CharacterEmoteEvent, ...] = ()
        replacement_parser: CharacterEmoteStreamParser | None = None
        if capture is not None:
            replacement_parser = CharacterEmoteStreamParser()
            pushed = replacement_parser.push(selected_body)
            flushed = replacement_parser.flush()
            selected_body = pushed.visible_text + flushed.visible_text
            replacement_events = pushed.events + flushed.events

        message.content = selected_body
        buffer = self._stream_chunks_by_message.get(message.id)
        if buffer is None:
            self._stream_chunks_by_message[message.id] = [selected_body]
        else:
            buffer[:] = [selected_body]
        self._stream_materialized_counts[message.id] = 1
        if capture is not None and replacement_parser is not None:
            capture.parser = replacement_parser
            capture.events = list(replacement_events)
            capture.fail_closed = False
            self._publish_character_emote_events(message.id, replacement_events)
        self._bump_message_speech_revision(message.id)
        self._bump_payload_revision(self._message_session_index[message.id])
        return self._snapshot(message)

    def set_citation_presentation(
        self,
        message_id: str,
        presentation: ConsoleCitationPresentation | None,
    ) -> ConsoleChatMessage:
        """Set content-free transient citation presentation for one message."""
        message = self._message_or_raise(message_id)
        if (
            presentation is not None
            and type(presentation) is not ConsoleCitationPresentation
        ):
            raise ValueError("presentation must be ConsoleCitationPresentation or None")
        message.citation_presentation = presentation
        return self._snapshot(message)

    def set_message_feedback(
        self,
        message_id: str,
        feedback: ConsoleMessageFeedback | None,
    ) -> ConsoleChatMessage:
        """Record user feedback on a complete Console message."""
        message = self._message_or_raise(message_id)
        self._materialize_stream_buffer(message)
        if message.status in {"pending", "streaming"}:
            raise ValueError("Wait for response to finish before recording feedback.")
        message.feedback = feedback
        self._persist_existing_message(message, update_feedback=True)
        return self._snapshot(message)

    def update_message_content(
        self, message_id: str, content: str
    ) -> ConsoleChatMessage:
        """Update a complete Console message or its currently selected variant."""
        if not content.strip():
            raise ValueError("Message content cannot be blank.")
        message = self._message_or_raise(message_id)
        self._materialize_stream_buffer(message)
        if message.status in {"pending", "streaming"}:
            raise ValueError("Wait for response to finish before editing this message.")
        session_id = self._message_session_index[message.id]
        previous_content = message.content
        if (
            message.role is ConsoleMessageRole.ASSISTANT
            and content != previous_content
            and not message.thinking_actions_enabled
        ):
            raise ConsoleThinkingCompatibilityError(
                "This conversation contains a newer thinking format; "
                "upgrade before editing it."
            )
        descendant_ids = self._subtree_ids(session_id, message.id)[1:]
        on_active_path = self._message_is_on_active_path(message.id)
        provenance_cleared = (
            message.metadata is not None
            and message.metadata.template_kind == "character_greeting"
        )
        if provenance_cleared:
            # An edit turns the row into ordinary user-owned content. Never
            # infer provenance later from matching text; clear it at the one
            # explicit ownership-transfer boundary.
            message.metadata = replace(
                message.metadata,
                template_kind="",
                template_source="",
            )
        if message.variants is None:
            message.content = content
        else:
            selected_index = message.variants.selected_index
            message.variants.variants[selected_index] = replace(
                message.variants.variants[selected_index],
                content=content,
                thinking=None,
                opaque_thinking_json=None,
                thinking_warning=None,
                thinking_actions_enabled=True,
                provider_continuation=None,
                provider_continuation_warning=None,
                provider_continuation_remote=False,
                provider_continuation_actions_enabled=True,
            )
            self._apply_generation_variant(message, message.variants.current)
        generation_cleared = (
            message.role is ConsoleMessageRole.ASSISTANT
            and message.content != previous_content
        )
        if generation_cleared:
            message.thinking = None
            message.opaque_thinking_json = None
            message.thinking_warning = None
            message.provider_continuation = None
            message.provider_continuation_warning = None
            message.provider_continuation_remote = False
            message.provider_continuation_message_version = None
            message.provider_continuation_actions_enabled = True
        self._bump_message_speech_revision(message.id)
        if provenance_cleared:
            self._bump_identity_revision(session_id)
        else:
            self._bump_payload_revision(session_id)
        if on_active_path and message.content != previous_content:
            self._bump_conversation_context_epoch(session_id)
        persisted = self._persist_existing_message(
            message,
            force_metadata_write=provenance_cleared,
            clear_generation_provenance=generation_cleared,
        )
        if persisted and message.content != previous_content and descendant_ids:
            self._purge_descendants_invalidated_by_edit(
                session_id, message.id, descendant_ids
            )
        if persisted and message.content != previous_content:
            self.record_trace_event(
                session_id,
                anchor_message_id=message.id,
                event_kind="message_edited",
                summary="Message edited",
                status="completed",
                source_event_id=(
                    f"message:{message.persisted_message_id}"
                    if message.persisted_message_id is not None
                    else None
                ),
            )
        return self._snapshot(message)

    def finalize_deferred_user_message_content(
        self, message_id: str, content: str
    ) -> ConsoleChatMessage:
        """Fill a blank user turn without invalidating replies already beneath it.

        Realtime input transcription can finish after the assistant reply has
        started. That is delayed completion of the original user turn, not an
        edit: descendants created while transcription was pending remain valid.

        Args:
            message_id: Native id of the initially blank user message.
            content: Final non-empty transcript text.

        Raises:
            ValueError: If the content or message is outside this narrow
                deferred-user-turn contract.
        """
        if type(content) is not str or not content.strip():
            raise ValueError("Deferred user message content must be non-empty text.")
        message = self._message_or_raise(message_id)
        if (
            message.role is not ConsoleMessageRole.USER
            or message.content
            or message.attachments
            or message.persisted_message_id is not None
        ):
            raise ValueError("Message is not a deferred blank user turn.")
        session_id = self._message_session_index[message.id]
        message.content = content
        self._bump_message_speech_revision(message.id)
        self._bump_payload_revision(session_id)
        if self._message_is_on_active_path(message.id):
            self._bump_conversation_context_epoch(session_id)
        self._persist_pending_message_if_ready(message)
        return self._snapshot(message)

    def _purge_descendants_invalidated_by_edit(
        self,
        session_id: str,
        owner_id: str,
        descendant_ids: Sequence[str],
    ) -> None:
        """Remove an edited node's stale descendants after the DB tombstones commit."""
        nodes = self._nodes_by_session.get(session_id, {})
        persisted_ids = [
            nodes[node_id].persisted_message_id
            for node_id in descendant_ids
            if node_id in nodes and nodes[node_id].persisted_message_id is not None
        ]
        database = getattr(self.persistence, "db", None) if self.persistence else None
        tombstone_reader = getattr(database, "get_message_tombstones", None)
        if callable(tombstone_reader):
            self._project_sync_v2_message_deletes(tombstone_reader(persisted_ids))

        children_map = self._children_by_parent.get(session_id, {})
        children_map.pop(owner_id, None)
        removed = set(descendant_ids)
        for node_id in descendant_ids:
            self.clear_terminal_citation_state(node_id)
            nodes.pop(node_id, None)
            children_map.pop(node_id, None)
            self._native_parent_by_message.pop(node_id, None)
            self._message_session_index.pop(node_id, None)
            self._stream_chunks_by_message.pop(node_id, None)
            self._stream_materialized_counts.pop(node_id, None)
            self._pending_persistence_message_ids.discard(node_id)
            self._variant_stream_bases.pop(node_id, None)
            self._variant_restored_message_ids.discard(node_id)
            self._failed_retry_message_ids.discard(node_id)
            self._message_speech_revisions.pop(node_id, None)
            self._exchange_blob_cache.pop(node_id, None)
            self._character_emote_captures.pop(node_id, None)
        self._purge_tool_markers(session_id, removed)
        if self._active_leaf_by_session.get(session_id) in removed:
            self._active_leaf_by_session[session_id] = owner_id
            self._persist_active_leaf(session_id, owner_id)
        self._recompute_active_path(session_id)

    def presentation_context(
        self, session_id: str, global_default: object
    ) -> ConsolePresentationContext:
        """Resolve live display identity for one session without storing a copy."""
        session = self._session_or_raise(session_id)
        return ConsolePresentationContext(
            user_name=effective_user_display_name(
                session.user_display_name_override, global_default
            ),
            assistant_kind=session.assistant_kind,
            character_name=session.character_name,
            revision=session.identity_revision,
        )

    def set_session_user_display_name_override(
        self,
        session_id: str,
        value: object,
        *,
        global_default: object,
    ) -> tuple[ConsoleChatSession, bool]:
        """Set a per-chat human name and rematerialize trusted projections."""
        session = self._session_or_raise(session_id)
        normalized = normalize_chat_display_name(value, blank_means_none=True)
        if session.user_display_name_override == normalized:
            return session, True
        session.user_display_name_override = normalized
        self._bump_identity_revision(session_id)
        context_persisted = self._persist_roleplay_context(session)
        persisted = self._materialize_roleplay_projections(
            session_id, global_default=global_default
        )
        if not context_persisted:
            persisted = False
        return session, persisted

    def refresh_session_roleplay_projections(
        self,
        session_id: str,
        *,
        global_default: object,
    ) -> bool:
        """Refresh trusted character projections when the effective name changes."""
        plan = self.prepare_session_roleplay_projection_refresh(
            session_id,
            global_default=global_default,
        )
        if plan is None:
            return True
        plan = self.rebase_roleplay_projection_plan_sync(plan)
        result = self.persist_roleplay_projection_plan(plan)
        self.accept_roleplay_projection_persistence_result(result)
        return result.persisted

    def prepare_session_roleplay_projection_refresh(
        self,
        session_id: str,
        *,
        global_default: object,
        force_persistence: bool = False,
    ) -> ConsoleRoleplayProjectionPersistencePlan | None:
        """Materialize live projections and snapshot their durable writes.

        This owner-thread seam is deliberately separate from
        :meth:`persist_roleplay_projection_plan`: callers may move only the
        returned frozen plan off-thread, never this store or its live session
        and message objects.
        """
        session = self._session_or_raise(session_id)
        projection_is_stale = self._roleplay_projection_is_stale(
            session, global_default
        )
        if not projection_is_stale and not force_persistence:
            return None
        if projection_is_stale:
            self._bump_identity_revision(session_id)
        return self._materialize_roleplay_projections_live(
            session_id,
            global_default=global_default,
            force_persistence=force_persistence,
        )

    def seed_character_roleplay(
        self,
        session_id: str,
        *,
        system_template: str,
        greeting_template: str,
        global_default: object,
    ) -> ConsoleChatMessage | None:
        """Seed trusted character system/greeting sources into a fresh session."""
        session = self._session_or_raise(session_id)
        source = (
            system_template
            if isinstance(system_template, str) and system_template.strip()
            else None
        )
        source_changed = session.character_system_template != source
        session.character_system_template = source
        if source_changed:
            self._bump_identity_revision(session_id)
        context_persisted = self._persist_roleplay_context(session)
        self._materialize_roleplay_projections(
            session_id, global_default=global_default
        )
        if not context_persisted:
            logger.warning("Failed to persist seeded Console roleplay context.")
        return self._append_character_greeting(
            session_id,
            greeting_template=greeting_template,
            global_default=global_default,
            source_changed=source_changed,
        )

    def swap_session_character_roleplay(
        self,
        session_id: str,
        *,
        character_name: str | None,
        system_template: str,
        greeting_template: str,
        global_default: object,
    ) -> tuple[ConsoleChatSession, ConsoleChatMessage | None, bool]:
        """Swap character identity and trusted sources before materializing once.

        The live mutation is intentionally not rolled back when a durable write
        fails. The returned status lets the UI surface that partial outcome
        without ever persisting an old-template/new-character projection.

        Args:
            session_id: Native Console session to rebind.
            character_name: New character display name.
            system_template: Trusted raw character system source.
            greeting_template: Trusted greeting source, or blank to skip seeding.
            global_default: Current global human display name.

        Returns:
            The live session, any newly seeded greeting, and whether every
            required durable write succeeded.
        """
        session = self._session_or_raise(session_id)
        normalized_name = (
            character_name.strip() if isinstance(character_name, str) else ""
        )
        new_name = normalized_name or None
        source = (
            system_template
            if isinstance(system_template, str) and system_template.strip()
            else None
        )
        identity_changed = (
            session.character_name != new_name
            or session.character_system_template != source
        )
        source_changed = session.character_system_template != source
        session.character_name = new_name
        session.character_system_template = source
        if identity_changed:
            self._bump_identity_revision(session_id)

        context_persisted = self._persist_roleplay_context(session)
        persisted = self._materialize_roleplay_projections(
            session_id,
            global_default=global_default,
        )
        if not context_persisted:
            persisted = False
        greeting = self._append_character_greeting(
            session_id,
            greeting_template=greeting_template,
            global_default=global_default,
            source_changed=source_changed,
        )
        if (
            greeting is not None
            and self.persistence is not None
            and not session.ephemeral
            and greeting.persisted_message_id is None
        ):
            persisted = False
        return session, greeting, persisted

    def _append_character_greeting(
        self,
        session_id: str,
        *,
        greeting_template: str,
        global_default: object,
        source_changed: bool,
    ) -> ConsoleChatMessage | None:
        """Append one trusted greeting projection when a source is present."""
        session = self._session_or_raise(session_id)
        if not isinstance(greeting_template, str) or not greeting_template.strip():
            return None
        context = self.presentation_context(session_id, global_default)
        greeting = expand_character_template(
            greeting_template,
            user_name=context.user_name,
            character_name=(session.character_name or "").strip(),
        )
        if not source_changed and any(
            message.role is ConsoleMessageRole.ASSISTANT
            and message.content == greeting
            and message.metadata is not None
            and message.metadata.template_kind == "character_greeting"
            and message.metadata.template_source == greeting_template
            for message in self._nodes_by_session.get(session_id, {}).values()
        ):
            return None
        return self.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=greeting,
            persist=True,
            metadata=MessageMetadata(
                template_kind="character_greeting",
                template_source=greeting_template,
            ),
        )

    def set_session_character_name(
        self,
        session_id: str,
        character_name: str | None,
        *,
        global_default: object,
    ) -> tuple[ConsoleChatSession, bool]:
        """Set character identity through the projection revision seam."""
        session = self._session_or_raise(session_id)
        normalized = character_name.strip() if isinstance(character_name, str) else ""
        new_name = normalized or None
        if session.character_name == new_name:
            return session, True
        session.character_name = new_name
        self._bump_identity_revision(session_id)
        persisted = self._materialize_roleplay_projections(
            session_id, global_default=global_default
        )
        return session, persisted

    def _bump_identity_revision(self, session_id: str) -> None:
        session = self._session_or_raise(session_id)
        session.identity_revision += 1
        self._bump_payload_revision(session_id)

    def _clear_character_greeting_provenance(self, message: ConsoleChatMessage) -> bool:
        """Revoke trusted greeting provenance after a generated replacement wins."""
        if (
            message.metadata is None
            or message.metadata.template_kind != "character_greeting"
        ):
            return False
        message.metadata = replace(
            message.metadata,
            template_kind="",
            template_source="",
        )
        self._bump_identity_revision(self._message_session_index[message.id])
        return True

    @staticmethod
    def _is_named_character_session(session: ConsoleChatSession) -> bool:
        return (
            session.assistant_kind == "character"
            and isinstance(session.character_name, str)
            and bool(session.character_name.strip())
        )

    def _roleplay_projection_is_stale(
        self, session: ConsoleChatSession, global_default: object
    ) -> bool:
        if not self._is_named_character_session(session):
            return False
        context = self.presentation_context(session.id, global_default)
        character_name = (session.character_name or "").strip()
        template = session.character_system_template
        if template and session.settings is not None:
            if session.settings.system_prompt != expand_character_template(
                template,
                user_name=context.user_name,
                character_name=character_name,
            ):
                return True
        for message in self._nodes_by_session.get(session.id, {}).values():
            metadata = message.metadata
            if (
                metadata is not None
                and metadata.template_kind == "character_greeting"
                and metadata.template_source.strip()
                and message.content
                != expand_character_template(
                    metadata.template_source,
                    user_name=context.user_name,
                    character_name=character_name,
                )
            ):
                return True
        return False

    def _materialize_roleplay_projections(
        self, session_id: str, *, global_default: object
    ) -> bool:
        """Materialize and synchronously persist trusted projections."""
        plan = self._materialize_roleplay_projections_live(
            session_id,
            global_default=global_default,
        )
        if plan is None:
            return True
        result = self.persist_roleplay_projection_plan(plan)
        self.accept_roleplay_projection_persistence_result(result)
        return result.persisted

    def _materialize_roleplay_projections_live(
        self,
        session_id: str,
        *,
        global_default: object,
        force_persistence: bool = False,
    ) -> ConsoleRoleplayProjectionPersistencePlan | None:
        """Update owner-thread state and return frozen durable call arguments."""
        session = self._session_or_raise(session_id)
        if not self._is_named_character_session(session):
            return None
        context = self.presentation_context(session_id, global_default)
        character_name = (session.character_name or "").strip()
        system_prompt_write: _RoleplaySystemPromptWrite | None = None
        message_writes: list[_RoleplayMessageProjectionWrite] = []
        if session.character_system_template and session.settings is not None:
            projected_system = expand_character_template(
                session.character_system_template,
                user_name=context.user_name,
                character_name=character_name,
            )
            if session.settings.system_prompt != projected_system:
                prior_system_prompt = session.settings.system_prompt
                session.settings = replace(
                    session.settings, system_prompt=projected_system
                )
                system_prompt_write = self._snapshot_roleplay_system_prompt_write(
                    session,
                    projected_system,
                    prior_system_prompt=prior_system_prompt,
                )
            elif force_persistence:
                system_prompt_write = self._snapshot_roleplay_system_prompt_write(
                    session,
                    projected_system,
                    prior_system_prompt=projected_system,
                    source_owned_repair=True,
                )
        for message in self._nodes_by_session.get(session_id, {}).values():
            metadata = message.metadata
            if (
                metadata is None
                or metadata.template_kind != "character_greeting"
                or not metadata.template_source.strip()
            ):
                continue
            projected = expand_character_template(
                metadata.template_source,
                user_name=context.user_name,
                character_name=character_name,
            )
            if message.content == projected and not force_persistence:
                continue
            prior_content = message.content
            if message.content != projected:
                message.content = projected
                if message.variants is not None:
                    selected = message.variants.selected_index
                    message.variants.variants[selected] = replace(
                        message.variants.variants[selected], content=projected
                    )
                self._bump_message_speech_revision(message.id)
            message_write = self._snapshot_roleplay_message_projection_write(
                session,
                message,
                prior_content=prior_content,
                source_owned_repair=force_persistence,
            )
            if message_write is not None:
                message_writes.append(message_write)
        if system_prompt_write is None and not message_writes:
            return None
        return ConsoleRoleplayProjectionPersistencePlan(
            session_id=session.id,
            generation=session.identity_revision,
            system_prompt_write=system_prompt_write,
            message_writes=tuple(message_writes),
        )

    def _snapshot_roleplay_system_prompt_write(
        self,
        session: ConsoleChatSession,
        system_prompt: str | None,
        *,
        prior_system_prompt: str | None,
        source_owned_repair: bool = False,
    ) -> _RoleplaySystemPromptWrite | None:
        if self.persistence is None or session.persisted_conversation_id is None:
            return None
        writer = getattr(self.persistence, "update_conversation_system_prompt", None)
        if not callable(writer):
            writer = _refuse_roleplay_projection_write
        accepts_roleplay_version_guard = self._persistence_accepts_kwarg(
            writer, "expected_roleplay_version"
        )
        expected_roleplay_version = None
        if source_owned_repair and accepts_roleplay_version_guard:
            reader = getattr(self.persistence, "get_conversation_version", None)
            if callable(reader):
                expected_roleplay_version = reader(session.persisted_conversation_id)
            if expected_roleplay_version is None:
                expected_roleplay_version = 0
        previous_candidates = self._roleplay_system_projection_candidates.get(
            session.id, ()
        )
        expected_system_prompts = tuple(
            dict.fromkeys((*previous_candidates, prior_system_prompt))
        )
        self._roleplay_system_projection_candidates[session.id] = tuple(
            dict.fromkeys((*expected_system_prompts, system_prompt))
        )
        return _RoleplaySystemPromptWrite(
            writer=writer,
            conversation_id=session.persisted_conversation_id,
            system_prompt=system_prompt,
            expected_roleplay_context=ConsoleRoleplayContext(
                user_name_override=session.user_display_name_override,
                character_system_template=session.character_system_template,
                character_name_snapshot=(
                    session.character_name
                    if session.assistant_kind == "character"
                    else None
                ),
            ),
            expected_system_prompts=expected_system_prompts,
            accepts_roleplay_context_guard=self._persistence_accepts_kwarg(
                writer, "expected_roleplay_context"
            ),
            accepts_system_prompt_guard=self._persistence_accepts_kwarg(
                writer, "expected_system_prompts"
            ),
            accepts_source_owned_repair=self._persistence_accepts_kwarg(
                writer, "allow_source_owned_repair"
            ),
            source_owned_repair=source_owned_repair,
            accepts_roleplay_version_guard=accepts_roleplay_version_guard,
            expected_roleplay_version=expected_roleplay_version,
        )

    def _snapshot_roleplay_message_projection_write(
        self,
        session: ConsoleChatSession,
        message: ConsoleChatMessage,
        *,
        prior_content: str,
        source_owned_repair: bool = False,
    ) -> _RoleplayMessageProjectionWrite | None:
        if self.persistence is None or message.persisted_message_id is None:
            return None
        writer = getattr(self.persistence, "update_message_content", None)
        if not callable(writer):
            return None
        accepts_roleplay_version_guard = self._persistence_accepts_kwarg(
            writer, "expected_roleplay_version"
        )
        expected_roleplay_version = None
        if source_owned_repair and accepts_roleplay_version_guard:
            reader = getattr(self.persistence, "get_message_version", None)
            if callable(reader):
                expected_roleplay_version = reader(message.persisted_message_id)
            if expected_roleplay_version is None:
                expected_roleplay_version = 0
        metadata_json = (
            message.metadata.to_json()
            if message.metadata is not None and not message.metadata.is_empty
            else None
        )
        metadata = message.metadata
        if metadata is None or metadata.template_kind != "character_greeting":
            return None
        previous_candidates = self._roleplay_message_projection_candidates.get(
            message.id, ()
        )
        expected_message_contents = tuple(
            dict.fromkeys((*previous_candidates, prior_content))
        )
        self._roleplay_message_projection_candidates[message.id] = tuple(
            dict.fromkeys((*expected_message_contents, message.content))
        )
        return _RoleplayMessageProjectionWrite(
            writer=writer,
            native_message_id=message.id,
            message_id=message.persisted_message_id,
            content=message.content,
            image_data=message.image_data,
            image_mime_type=message.image_mime_type,
            feedback=message.feedback,
            metadata_json=metadata_json,
            accepts_attachments=self._persistence_accepts_kwarg(writer, "attachments"),
            accepts_metadata_json=self._persistence_accepts_kwarg(
                writer, "metadata_json"
            ),
            expected_roleplay_template_source=metadata.template_source,
            expected_message_contents=expected_message_contents,
            accepts_template_source_guard=self._persistence_accepts_kwarg(
                writer, "expected_roleplay_template_source"
            ),
            accepts_message_contents_guard=self._persistence_accepts_kwarg(
                writer, "expected_message_contents"
            ),
            accepts_source_owned_repair=self._persistence_accepts_kwarg(
                writer, "allow_source_owned_repair"
            ),
            source_owned_repair=source_owned_repair,
            accepts_roleplay_version_guard=accepts_roleplay_version_guard,
            expected_roleplay_version=expected_roleplay_version,
            sync_write=self._snapshot_roleplay_sync_write(session, message),
        )

    def is_roleplay_projection_plan_current(
        self, plan: ConsoleRoleplayProjectionPersistencePlan
    ) -> bool:
        """Return whether a queued plan still owns the session generation."""
        session = self._sessions.get(plan.session_id)
        return session is not None and session.identity_revision == plan.generation

    def rebase_roleplay_projection_plan_sync(
        self, plan: ConsoleRoleplayProjectionPersistencePlan
    ) -> ConsoleRoleplayProjectionPersistencePlan:
        """Rebase frozen Sync writes from the latest serialized predecessor."""
        rebased_messages: list[_RoleplayMessageProjectionWrite] = []
        for message_write in plan.message_writes:
            sync_write = message_write.sync_write
            if sync_write is None:
                rebased_messages.append(message_write)
                continue
            kwargs = dict(sync_write.kwargs)
            kwargs["base_version"] = self._sync_v2_message_versions.get(
                sync_write.stable_key
            )
            rebased_messages.append(
                replace(
                    message_write,
                    sync_write=replace(sync_write, kwargs=tuple(kwargs.items())),
                )
            )
        return replace(plan, message_writes=tuple(rebased_messages))

    def _snapshot_roleplay_sync_write(
        self,
        session: ConsoleChatSession,
        message: ConsoleChatMessage,
    ) -> _RoleplaySyncWrite | None:
        if (
            self.sync_v2_chat_producer is None
            or self.sync_v2_server_profile_id is None
            or session.persisted_conversation_id is None
            or message.persisted_message_id is None
            or message.status != "complete"
            or not message.content
        ):
            return None
        writer = getattr(self.sync_v2_chat_producer, "enqueue_chat_message", None)
        if not callable(writer):
            return None
        variant_metadata = self._sync_variant_metadata(message)
        stable_key = (
            f"{session.persisted_conversation_id}:{message.persisted_message_id}"
        )
        kwargs = {
            "server_profile_id": self.sync_v2_server_profile_id,
            "authenticated_principal_id": self.sync_v2_authenticated_principal_id,
            "workspace_scope": self.sync_v2_workspace_scope,
            "conversation_id": session.persisted_conversation_id,
            "message_id": message.persisted_message_id,
            "role": message.role.value,
            "content": message.content,
            "parent_message_id": self._previous_persisted_message_id(message),
            "sequence": self._sync_message_sequence(message),
            "variant_turn_id": variant_metadata["variant_turn_id"],
            "variant_index": variant_metadata["variant_index"],
            "variant_count": variant_metadata["variant_count"],
            "selected_variant_id": variant_metadata["selected_variant_id"],
            "base_version": self._sync_v2_message_versions.get(stable_key),
            "entity_version": None,
        }
        return _RoleplaySyncWrite(
            writer=writer,
            stable_key=stable_key,
            kwargs=tuple(kwargs.items()),
        )

    @staticmethod
    def persist_roleplay_projection_plan(
        plan: ConsoleRoleplayProjectionPersistencePlan,
    ) -> ConsoleRoleplayProjectionPersistenceResult:
        """Consume one frozen plan without reading or mutating a live store."""
        persisted = True
        system_write = plan.system_prompt_write
        system_prompt_persisted = True
        if system_write is not None:
            try:
                system_kwargs: dict[str, object] = {
                    "conversation_id": system_write.conversation_id,
                    "system_prompt": system_write.system_prompt,
                }
                if system_write.accepts_roleplay_context_guard:
                    system_kwargs["expected_roleplay_context"] = (
                        system_write.expected_roleplay_context
                    )
                if system_write.accepts_system_prompt_guard:
                    system_kwargs["expected_system_prompts"] = (
                        system_write.expected_system_prompts
                    )
                if system_write.accepts_source_owned_repair:
                    system_kwargs["allow_source_owned_repair"] = (
                        system_write.source_owned_repair
                    )
                if system_write.accepts_roleplay_version_guard:
                    system_kwargs["expected_roleplay_version"] = (
                        system_write.expected_roleplay_version
                    )
                if not system_write.writer(
                    **system_kwargs,
                ):
                    system_prompt_persisted = False
                    persisted = False
            except Exception as exc:
                system_prompt_persisted = False
                persisted = False
                logger.warning(
                    "Failed to persist planned Console roleplay system prompt "
                    "projection (error_type={}).",
                    type(exc).__name__,
                )
        message_outcomes: list[_RoleplayMessageProjectionPersistenceOutcome] = []
        for message_write in plan.message_writes:
            kwargs: dict[str, object] = {
                "message_id": message_write.message_id,
                "content": message_write.content,
                "image_data": message_write.image_data,
                "image_mime_type": message_write.image_mime_type,
                "parent_message_id": None,
                "feedback": message_write.feedback,
                "update_parent": False,
                "update_feedback": False,
            }
            if message_write.accepts_attachments:
                kwargs["attachments"] = None
            if (
                message_write.accepts_metadata_json
                and message_write.metadata_json is not None
            ):
                kwargs["metadata_json"] = message_write.metadata_json
            if message_write.accepts_template_source_guard:
                kwargs["expected_roleplay_template_source"] = (
                    message_write.expected_roleplay_template_source
                )
            if message_write.accepts_message_contents_guard:
                kwargs["expected_message_contents"] = (
                    message_write.expected_message_contents
                )
            if message_write.accepts_source_owned_repair:
                kwargs["allow_source_owned_repair"] = message_write.source_owned_repair
            if message_write.accepts_roleplay_version_guard:
                kwargs["expected_roleplay_version"] = (
                    message_write.expected_roleplay_version
                )
            try:
                message_persisted = bool(message_write.writer(**kwargs))
            except Exception as exc:
                message_persisted = False
                logger.warning(
                    "Failed to persist planned Console roleplay message projection "
                    "(error_type={}).",
                    type(exc).__name__,
                )
            if not message_persisted:
                persisted = False
            message_outcomes.append(
                _RoleplayMessageProjectionPersistenceOutcome(
                    native_message_id=message_write.native_message_id,
                    content=message_write.content,
                    persisted=message_persisted,
                    sync_write=(
                        message_write.sync_write if message_persisted else None
                    ),
                )
            )
        return ConsoleRoleplayProjectionPersistenceResult(
            session_id=plan.session_id,
            generation=plan.generation,
            persisted=persisted,
            system_prompt_attempted=system_write is not None,
            system_prompt=(
                system_write.system_prompt if system_write is not None else None
            ),
            system_prompt_persisted=system_prompt_persisted,
            message_outcomes=tuple(message_outcomes),
        )

    def accept_roleplay_projection_persistence_result(
        self,
        result: ConsoleRoleplayProjectionPersistenceResult,
    ) -> bool:
        """Apply completion bookkeeping only for the still-current generation."""
        session = self._sessions.get(result.session_id)
        if session is None or session.identity_revision != result.generation:
            return False
        if result.system_prompt_attempted and result.system_prompt_persisted:
            self._roleplay_system_projection_candidates[session.id] = (
                result.system_prompt,
            )
        for outcome in result.message_outcomes:
            if not outcome.persisted:
                continue
            self._roleplay_message_projection_candidates[outcome.native_message_id] = (
                outcome.content,
            )
            self._enqueue_accepted_roleplay_sync(outcome)
        return True

    def _enqueue_accepted_roleplay_sync(
        self, outcome: _RoleplayMessageProjectionPersistenceOutcome
    ) -> None:
        """Emit Sync only after this owner accepts the projection generation."""
        sync_write = outcome.sync_write
        if sync_write is None:
            return
        kwargs = dict(sync_write.kwargs)
        kwargs["base_version"] = self._sync_v2_message_versions.get(
            sync_write.stable_key
        )
        try:
            sync_result = sync_write.writer(**kwargs)
            if not (
                isinstance(sync_result, dict)
                and sync_result.get("status") == "enqueued"
            ):
                return
            entry = sync_result.get("outbox_entry")
            envelope = entry.get("envelope") if isinstance(entry, dict) else None
            payload_hash = (
                envelope.get("payload_hash") if isinstance(envelope, dict) else None
            )
            if isinstance(payload_hash, str) and payload_hash:
                self._sync_v2_message_versions[sync_write.stable_key] = payload_hash
        except Exception as exc:
            logger.warning(
                "Failed to enqueue roleplay Sync v2 chat message after local mutation "
                "(error_type={}).",
                type(exc).__name__,
            )

    def _persist_message_projection(self, message: ConsoleChatMessage) -> bool:
        if self.persistence is None or message.persisted_message_id is None:
            return True
        try:
            return self._persist_existing_message(message)
        except Exception as exc:
            logger.warning(
                "Failed to persist Console roleplay message projection "
                "(error_type={}).",
                type(exc).__name__,
            )
            return False

    def _persist_session_system_prompt(
        self, session: ConsoleChatSession, system_prompt: str | None
    ) -> bool:
        if self.persistence is None or session.persisted_conversation_id is None:
            return True
        writer = getattr(self.persistence, "update_conversation_system_prompt", None)
        if not callable(writer):
            return False
        try:
            return bool(
                writer(
                    conversation_id=session.persisted_conversation_id,
                    system_prompt=system_prompt,
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist Console roleplay system prompt projection "
                "(error_type={}).",
                type(exc).__name__,
            )
            return False

    def _persist_roleplay_context(self, session: ConsoleChatSession) -> bool:
        if self.persistence is None or session.persisted_conversation_id is None:
            return True
        writer = getattr(self.persistence, "update_conversation_roleplay_context", None)
        if not callable(writer):
            return False
        try:
            return bool(
                writer(
                    conversation_id=session.persisted_conversation_id,
                    user_name_override=session.user_display_name_override,
                    character_system_template=session.character_system_template,
                    character_name_snapshot=(
                        session.character_name
                        if session.assistant_kind == "character"
                        else None
                    ),
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist Console roleplay identity context (error_type={}).",
                type(exc).__name__,
            )
            return False

    def delete_message(self, message_id: str) -> ConsoleChatMessage:
        """Durably tombstone a complete Console message and its subtree."""
        message = self._message_or_raise(message_id)
        self._materialize_stream_buffer(message)
        if message.status in {"pending", "streaming"}:
            raise ValueError(
                "Wait for response to finish before deleting this message."
            )
        session_id = self._message_session_index[message_id]
        parent_native_id = self._native_parent_by_message.get(message_id)
        on_active_path = message_id in self.active_path_message_ids(session_id)
        subtree_ids = self._subtree_ids(session_id, message_id)
        tombstones: list[dict[str, Any]] = []
        if self.persistence is not None and message.persisted_message_id is not None:
            deleter = getattr(self.persistence, "delete_message_subtree", None)
            if not callable(deleter):
                raise RuntimeError("Message deletion could not be persisted.")
            tombstones = deleter(message_id=message.persisted_message_id)
            self._project_sync_v2_message_deletes(tombstones)
        children_map = self._children_by_parent.get(session_id, {})
        nodes = self._nodes_by_session.get(session_id, {})
        # Detach the deleted node from its parent's ordered child list.
        siblings = children_map.get(parent_native_id)
        if siblings is not None and message_id in siblings:
            siblings.remove(message_id)
            if not siblings:
                children_map.pop(parent_native_id, None)
        # Purge the deleted node AND its whole subtree from every structure --
        # deleting a mid-conversation node drops the branch beneath it.
        for node_id in subtree_ids:
            self.clear_terminal_citation_state(node_id)
            nodes.pop(node_id, None)
            children_map.pop(node_id, None)
            self._native_parent_by_message.pop(node_id, None)
            self._message_session_index.pop(node_id, None)
            self._stream_chunks_by_message.pop(node_id, None)
            self._stream_materialized_counts.pop(node_id, None)
            self._pending_persistence_message_ids.discard(node_id)
            self._variant_stream_bases.pop(node_id, None)
            self._variant_restored_message_ids.discard(node_id)
            self._failed_retry_message_ids.discard(node_id)
            self._message_speech_revisions.pop(node_id, None)
            self._message_completion_generations.pop(node_id, None)
            self._exchange_blob_cache.pop(node_id, None)
        # Only when the deleted branch was on the active path does the leaf move
        # (up to the deleted node's parent); an off-path delete leaves it alone.
        self._purge_tool_markers(session_id, set(subtree_ids))
        if on_active_path:
            self._active_leaf_by_session[session_id] = parent_native_id
            self._persist_active_leaf(session_id, parent_native_id)
        self._recompute_active_path(session_id)
        self._bump_payload_revision(session_id)
        if on_active_path:
            self._bump_conversation_context_epoch(session_id)
        return self._snapshot(message)

    def session_id_for_message(self, message_id: str) -> str:
        """Return the owning session ID for a message."""
        if message_id not in self._message_session_index:
            raise KeyError(f"Unknown Console message: {message_id}")
        return self._message_session_index[message_id]

    def interrupted_provider_continuation_message(
        self,
        session_id: str | None = None,
    ) -> ConsoleChatMessage | None:
        """Return the active-path owner needing explicit recovery, if any."""
        target_session_id = session_id or self.active_session_id
        if target_session_id is None or target_session_id not in self._sessions:
            return None
        for message in reversed(self.messages_for_session(target_session_id)):
            checkpoint = message.provider_continuation
            if checkpoint is not None and checkpoint.state == "active":
                return message
        return None

    def provider_continuation_recovery_message(
        self,
        session_id: str | None = None,
    ) -> ConsoleChatMessage | None:
        """Return an active owner or safe warning for transcript recovery UI."""
        target_session_id = session_id or self.active_session_id
        if target_session_id is None or target_session_id not in self._sessions:
            return None
        for message in reversed(self.messages_for_session(target_session_id)):
            if message.provider_continuation_warning:
                return message
            checkpoint = message.provider_continuation
            if checkpoint is not None and checkpoint.state == "active":
                return message
        return None

    def set_provider_continuation_warning(
        self,
        message_id: str,
        warning: str,
    ) -> None:
        """Set bounded visible recovery copy without exposing private state."""
        self._message_or_raise(message_id).provider_continuation_warning = warning

    def discard_provider_continuation(
        self,
        message_id: str,
        *,
        expected_message_version: int,
    ) -> bool:
        """Optimistically clear one whole checkpoint without running tools."""
        message = self._message_or_raise(message_id)
        persisted_id = message.persisted_message_id
        database = getattr(self.persistence, "db", None) if self.persistence else None
        updater = getattr(database, "update_provider_continuation", None)
        if persisted_id is None or not callable(updater):
            raise RuntimeError(
                "Interrupted run could not be discarded; reload and retry."
            )
        session_id = self._message_session_index[message_id]
        children = self._children_by_parent.get(session_id, {})
        if (
            not message.content
            and not message.attachments
            and message.image_data is None
            and children.get(message_id)
        ):
            raise RuntimeError("Interrupted run changed; reload before discarding.")
        updater(
            message_id=persisted_id,
            expected_message_version=expected_message_version,
            provider_continuation_json=None,
            assistant_generation_state="discarded",
        )
        message.provider_continuation = None
        message.provider_continuation_message_version = expected_message_version + 1
        message.provider_continuation_remote = False
        message.provider_continuation_warning = None
        message.provider_continuation_actions_enabled = False
        message.assistant_generation_state = "discarded"
        if message.content or message.attachments or message.image_data is not None:
            self._refresh_and_project_provider_continuation(message)
            self._bump_payload_revision(session_id)
            return True

        self._project_sync_v2_message_deletes(
            (
                {
                    "message_id": persisted_id,
                    "version": expected_message_version + 1,
                },
            )
        )

        parent_id = self._native_parent_by_message.pop(message_id, None)
        siblings = children.get(parent_id, [])
        if message_id in siblings:
            siblings.remove(message_id)
        if not siblings:
            children.pop(parent_id, None)
        self._nodes_by_session.get(session_id, {}).pop(message_id, None)
        self._message_session_index.pop(message_id, None)
        self._pending_persistence_message_ids.discard(message_id)
        if self._active_leaf_by_session.get(session_id) == message_id:
            self._active_leaf_by_session[session_id] = parent_id
            self._persist_active_leaf(session_id, parent_id)
        self._recompute_active_path(session_id)
        self._bump_payload_revision(session_id)
        self._bump_conversation_context_epoch(session_id)
        return True

    def active_leaf(self, session_id: str) -> str | None:
        """Return the native id of the session's active-leaf node (or ``None``)."""
        self._session_or_raise(session_id)
        return self._active_leaf_by_session.get(session_id)

    def set_active_leaf(self, session_id: str, message_id: str | None) -> None:
        """Point a session's active leaf at a node and recompute the active path.

        Updates the in-memory pointer, rematerializes the active-path view via
        the single writer, and -- when the session owns a persisted conversation
        and the persistence adapter exposes a raw ``db`` seam -- write-throughs
        the local-only ``conversations.active_leaf_message_id`` pointer (mapped
        to the leaf node's *persisted* id, or ``None`` when the leaf is cleared
        or not yet persisted). A durable write failure is logged, never raised:
        the in-memory pointer is authoritative and already updated, matching
        this store's persist-through convention elsewhere.

        Args:
            session_id: Native Console session ID.
            message_id: Native id of the node to make the active leaf, or
                ``None`` to clear the active path entirely.

        Raises:
            KeyError: If the session is unknown, or ``message_id`` is not
                ``None`` and does not reference a node in the session's tree.
        """
        self._session_or_raise(session_id)
        nodes = self._nodes_by_session.get(session_id, {})
        if message_id is not None and message_id not in nodes:
            raise KeyError(f"Unknown Console message: {message_id}")
        previous_leaf = self._active_leaf_by_session.get(session_id)
        self._active_leaf_by_session[session_id] = message_id
        self._recompute_active_path(session_id)
        self._persist_active_leaf(session_id, message_id)
        self._bump_payload_revision(session_id)
        if message_id != previous_leaf:
            self._bump_conversation_context_epoch(session_id)
            if message_id is not None:
                self.record_trace_event(
                    session_id,
                    anchor_message_id=message_id,
                    event_kind="branch_selected",
                    summary="Conversation branch selected",
                    status="selected",
                    source_event_id=(
                        f"message:{nodes[previous_leaf].persisted_message_id}"
                        if previous_leaf in nodes
                        and nodes[previous_leaf].persisted_message_id is not None
                        else None
                    ),
                )

    def session_context_summary(self, session_id: str) -> tuple[str | None, str | None]:
        """Return the session's in-memory ``(summary, boundary_native_id)`` pair.

        Console `/rewind` "summarize up to here" (SP2). ``(None, None)`` when
        no summary has been set (including a freshly created session).
        """
        self._session_or_raise(session_id)
        return self._context_summary_by_session.get(session_id, (None, None))

    def set_session_context_summary(
        self,
        session_id: str,
        summary: str | None,
        boundary_native_id: str | None,
    ) -> None:
        """Set (or clear, with ``(None, None)``) a session's boundary summary.

        Updates the in-memory ``(summary, boundary_native_id)`` pair, then --
        when the session owns a persisted conversation and the persistence
        adapter exposes a raw ``db`` seam -- write-throughs the local-only
        ``conversations.context_summary`` / ``summary_boundary_message_id``
        columns (mapped to the boundary node's *persisted* id, or ``None``
        when the boundary is cleared or not yet persisted). A durable write
        failure is logged, never raised: the in-memory pair is authoritative
        and already updated, matching ``set_active_leaf``'s persist-through
        convention. Unlike ``set_active_leaf``, an unknown
        ``boundary_native_id`` does not raise -- it write-throughs a ``None``
        persisted id (treated as "not yet persisted"), matching the design's
        fail-open handling of a dangling boundary.

        Args:
            session_id: Native Console session ID.
            summary: The boundary summary text, or ``None`` to clear it.
            boundary_native_id: Native id of the message the summary covers
                up to, or ``None`` to clear it.

        Raises:
            KeyError: If the session is unknown.
        """
        self._session_or_raise(session_id)
        previous = self._context_summary_by_session.get(session_id, (None, None))
        updated = (summary, boundary_native_id)
        self._context_summary_by_session[session_id] = updated
        self._persist_context_summary(session_id, summary, boundary_native_id)
        self._bump_payload_revision(session_id)
        if updated != previous:
            self._bump_conversation_context_epoch(session_id)

    def active_path_message_ids(self, session_id: str) -> list[str]:
        """Return native ids along the active path, root -> active leaf.

        A visited-set guards against a malformed cyclic parent chain (real
        DBs can't produce one -- unique PKs -- so this is defensive-only,
        mirroring ``_nearest_persisted_ancestor_id``).

        Args:
            session_id: Native Console session ID.

        Returns:
            Native message ids on the active path, ordered root first and
            active leaf last; empty when the session has no active leaf.

        Raises:
            KeyError: If the session is unknown.
        """
        self._session_or_raise(session_id)
        ids: list[str] = []
        visited: set[str] = set()
        current = self._active_leaf_by_session.get(session_id)
        while current is not None and current not in visited:
            visited.add(current)
            ids.append(current)
            current = self._native_parent_by_message.get(current)
        ids.reverse()
        return ids

    def siblings_at(self, message_id: str) -> tuple[list[ConsoleChatMessage], int, int]:
        """Return ``(ordered sibling snapshots, index of message_id, count)``.

        Siblings are the children of ``message_id``'s native parent, in creation
        order. Snapshots are independent copies so callers cannot mutate the
        live tree nodes. Resolves from the full tree, so it works for off-path
        nodes too.

        Raises:
            KeyError: If ``message_id`` is not a node in any session's tree.
        """
        session_id = self._message_session_index.get(message_id)
        nodes = self._nodes_by_session.get(session_id or "", {})
        if session_id is None or message_id not in nodes:
            raise KeyError(f"Unknown Console message: {message_id}")
        parent_native_id = self._native_parent_by_message.get(message_id)
        sibling_ids = self._children_by_parent.get(session_id, {}).get(
            parent_native_id, []
        )
        snapshots = [self._snapshot(nodes[sibling_id]) for sibling_id in sibling_ids]
        index = sibling_ids.index(message_id) if message_id in sibling_ids else 0
        return snapshots, index, len(sibling_ids)

    def begin_character_emote_capture(
        self,
        message_id: str,
        snapshot: CharacterEmoteRunSnapshot,
    ) -> None:
        """Arm one character-owned assistant row for safe directive parsing."""

        message = self._message_or_raise(message_id)
        session_id = self._message_session_index[message_id]
        session = self._session_or_raise(session_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages support character emotes.")
        if message.status not in {"pending", "streaming"}:
            raise ValueError("Character emotes require an active assistant message.")
        if session.assistant_kind != "character":
            raise ValueError("Character emotes require character session ownership.")
        if not isinstance(snapshot, CharacterEmoteRunSnapshot):
            raise TypeError("snapshot must be CharacterEmoteRunSnapshot")
        if (
            snapshot.actor_id is not None
            and session.character_id is not None
            and snapshot.actor_id != session.character_id
        ):
            raise ValueError("Character emote snapshot actor does not own the session.")
        self._character_emote_captures[message_id] = _CharacterEmoteCapture(
            parser=CharacterEmoteStreamParser(),
            snapshot=snapshot,
        )

    def character_emote_events_after(
        self,
        session_id: str,
        cursor: int,
    ) -> tuple[CharacterEmoteLiveEvent, ...]:
        """Return ordered content-free live events newer than ``cursor``."""

        self._session_or_raise(session_id)
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a nonnegative integer")
        return tuple(
            event
            for event in self._character_emote_feed_by_session.get(session_id, ())
            if event.sequence > cursor
        )

    def _publish_character_emote_events(
        self,
        message_id: str,
        events: Sequence[CharacterEmoteEvent],
    ) -> None:
        session_id = self._message_session_index[message_id]
        feed = self._character_emote_feed_by_session.setdefault(
            session_id,
            deque(maxlen=512),
        )
        for event in events:
            self._character_emote_sequence += 1
            feed.append(
                CharacterEmoteLiveEvent(
                    sequence=self._character_emote_sequence,
                    session_id=session_id,
                    message_id=message_id,
                    state=event.state,
                )
            )

    def append_stream_chunk(self, message_id: str, chunk: str) -> ConsoleChatMessage:
        """Append streamed assistant content to an existing message.

        A chunk arriving for a message already ``"stopped"`` is dropped
        silently rather than raising: the user's Stop already finalized and
        persisted this message, so a late chunk from a slow provider (one
        that hadn't produced a single token before Stop was clicked) is
        benign by definition, not a programming error (Plan-B agent-runtime
        gate Finding 1 -- see ``Docs/superpowers/qa/agent-runtime-2026-07/
        README.md``). Other invalid statuses (``complete``/``failed``)
        still raise via ``_validate_can_stream`` -- those really do
        indicate a bug in the caller.
        """
        message = self._message_or_raise(message_id)
        if message.status == "stopped":
            return self._snapshot(message)
        self._validate_can_stream(message)
        capture = self._character_emote_captures.get(message.id)
        accepted_events: tuple[CharacterEmoteEvent, ...] = ()
        if capture is not None:
            checkpoint = capture.parser.safe_copy()
            try:
                parsed = capture.parser.push(chunk)
                if not capture.fail_closed:
                    accepted_events = parsed.events
            except Exception:
                logger.warning("character_emote_parser_failed")
                capture.fail_closed = True
                capture.parser = checkpoint
                parsed = capture.parser.push(chunk)
            chunk = parsed.visible_text
            if accepted_events:
                capture.events.extend(accepted_events)
                self._publish_character_emote_events(message.id, accepted_events)
        buffer = self._stream_chunks_by_message.setdefault(
            message.id,
            [message.content] if message.content else [],
        )
        if chunk:
            buffer.append(chunk)
        message.status = "streaming"
        if chunk:
            self._bump_message_speech_revision(message.id)
        # Trajectory sidecar (schema v38): the first provider chunk is the
        # first-token boundary. Stamped at THIS seam (rather than in the
        # controller's direct-provider loop) because it is the single point
        # both streaming paths -- direct provider and agent bridge -- flow
        # through. Only stamps an armed capture: no step-start, no timing.
        stash = self._trajectory_timing.get(message.id)
        if (
            chunk
            and stash is not None
            and stash.get("step_started_at") is not None
            and stash.get("first_token_at") is None
        ):
            stash["first_token_at"] = time.time()
        return self._snapshot(message)

    # --- Trajectory sidecar (schema v38) -----------------------------------
    #
    # LOCAL-ONLY capture of per-turn step observations for the Console
    # trajectory view. All methods on this seam are best-effort: a sidecar
    # failure is logged with context and NEVER fails the turn/chat action
    # that triggered it.

    def record_trajectory_timing(
        self,
        message_id: str,
        *,
        step_started_at: float | None = None,
        first_token_at: float | None = None,
        completed_at: float | None = None,
        model: str | None = None,
        provider: str | None = None,
        model_status: str | None = None,
        flush: bool = False,
    ) -> None:
        """Merge timing facts for one message's trajectory capture; never raises.

        ``step_started_at``/``first_token_at``/``model``/``provider`` are
        set-once (first writer wins); ``completed_at`` is last-write-wins so
        a finalize path can refine a provisional completion stamp. With
        ``flush=True``, a message that is already persisted gets its
        assistant/user sidecar row written immediately (the finalize path --
        usage attachment time); an unpersisted message's row is written by
        the persist seam instead, reading this same stash.
        """
        try:
            stash = self._trajectory_timing.setdefault(message_id, {})
            for key, value in (
                ("step_started_at", step_started_at),
                ("first_token_at", first_token_at),
                ("model", model),
                ("provider", provider),
            ):
                if value is not None and stash.get(key) is None:
                    stash[key] = value
            if completed_at is not None:
                stash["completed_at"] = completed_at
                stash.setdefault("model_status", "completed")
            if model_status is not None:
                stash["model_status"] = model_status
            if not flush:
                return
            message = self._nodes_lookup(message_id)
            if message is not None:
                self._write_trajectory_row_for_message(message)
        except Exception as exc:
            logger.bind(message_id=message_id, error=repr(exc)).warning(
                "trajectory_timing_record_failed"
            )

    def write_trajectory_rows(self, rows: Sequence[TrajectoryRowWrite]) -> bool:
        """Route sidecar rows through the persistence adapter; never raises.

        Serialized by an in-process lock so concurrent Console writers
        (hands-free sessions, compaction auxiliary turns) cannot interleave
        ``seq`` assignments. Fakes without the adapter method are skipped
        silently (pre-existing test doubles keep working).
        """
        if self.persistence is None or not rows:
            return False
        writer = getattr(self.persistence, "write_trajectory_rows", None)
        if not callable(writer):
            return False
        with self._trajectory_lock:
            try:
                result = writer(list(rows))
            except Exception:
                logger.warning("trajectory_rows_write_failed")
                result = False
            if result is False:
                self._write_capture_failed_diagnostic(writer, rows)
                return False
        if result is not False:
            # task-5: a successful sidecar write is trajectory-visible state.
            # Bump the revision bus (conversation key + every live session
            # bound to that conversation) so the polling trajectory screen
            # rebuilds its snapshot; without this, tail-follow sees nothing.
            for conversation_id in dict.fromkeys(row.conversation_id for row in rows):
                self._bump_payload_revision(conversation_id)
                for session in self._sessions.values():
                    if session.persisted_conversation_id == conversation_id:
                        self._bump_payload_revision(session.id)
        return result is not False

    def _write_capture_failed_diagnostic(
        self,
        writer: Callable[[Sequence[TrajectoryRowWrite]], object],
        rows: Sequence[TrajectoryRowWrite],
    ) -> None:
        """Attempt one payload-free diagnostic without re-entering capture."""
        first = rows[0]
        if any(row.event_kind == "capture_failed" for row in rows):
            return
        source_events: list[dict[str, str]] = []
        for row in rows:
            source_event_id = ""
            try:
                payload = json.loads(row.payload_json or "{}")
                if isinstance(payload, dict):
                    source_event_id = str(payload.get("event_id") or "")
            except (TypeError, ValueError):
                pass
            if not source_event_id:
                source_event_id = canonical_payload_hash(
                    {
                        "event_kind": row.event_kind,
                        "payload_json": row.payload_json or "",
                    }
                )
            source_events.append(
                {"event_kind": row.event_kind, "event_id": source_event_id}
            )
        digest = canonical_payload_hash(
            {
                "stage": "trajectory_write",
                "conversation_id": first.conversation_id,
                "message_id": first.message_id,
                "turn_id": first.turn_id,
                "source_events": source_events,
            }
        )
        event_id = f"capture-failed:{digest.removeprefix('sha256:')}"
        if event_id in self._trajectory_capture_failure_keys:
            return
        if first.conversation_id not in self._trajectory_capture_failure_hydrated:
            try:
                db = getattr(self.persistence, "db", None)
                reader = getattr(db, "get_trajectory_rows", None)
                if callable(reader):
                    for existing in reader(first.conversation_id):
                        if existing.event_kind != "capture_failed":
                            continue
                        try:
                            existing_payload = json.loads(existing.payload_json or "{}")
                        except (TypeError, ValueError):
                            continue
                        existing_id = existing_payload.get("event_id")
                        if isinstance(existing_id, str):
                            self._trajectory_capture_failure_keys.add(existing_id)
                    self._trajectory_capture_failure_hydrated.add(first.conversation_id)
            except Exception:  # noqa: BLE001 — diagnostic lookup is best-effort
                logger.warning("trajectory_capture_diagnostic_lookup_failed")
        if event_id in self._trajectory_capture_failure_keys:
            return
        self._trajectory_capture_failure_keys.add(event_id)
        diagnostic = TrajectoryRowWrite(
            message_id=first.message_id,
            conversation_id=first.conversation_id,
            turn_id=first.turn_id,
            seq=None,
            event_kind="capture_failed",
            step_started_at=first.step_started_at,
            payload_json=json.dumps(
                {
                    "event_id": event_id,
                    "summary": "Trace capture failed",
                    "status": "incomplete",
                    "field_states": {"payload": "capture_failed"},
                    "sensitivity": "diagnostic",
                }
            ),
        )
        try:
            writer([diagnostic])
        except Exception:
            logger.warning("trajectory_capture_diagnostic_write_failed")

    def variant_sets_for_conversation(
        self, conversation_id: str
    ) -> list[ConsoleVariantSet]:
        """Collect the live in-memory variant sets for one conversation.

        Variant CONTENTS are process-local (only selection metadata
        persists), so this covers sessions currently open in THIS store; a
        cold conversation restored purely from the DB legitimately
        contributes no variant contents and the trajectory ledger renders
        without superseded variants. Deduplicated per ``turn_id``, newest
        selection state last (the projection over-attaches a set to every
        assistant record of its turn; duplicates would double the contents).
        """
        sets_by_turn: dict[str, ConsoleVariantSet] = {}
        for session in self._sessions.values():
            if session.persisted_conversation_id != conversation_id:
                continue
            for message in self._nodes_by_session.get(session.id, {}).values():
                variants = getattr(message, "variants", None)
                if variants is None:
                    continue
                turn_id = getattr(variants, "turn_id", None) or getattr(
                    message, "turn_id", None
                )
                if not turn_id:
                    continue
                sets_by_turn[str(turn_id)] = variants
        return list(sets_by_turn.values())

    def _nodes_lookup(self, message_id: str) -> ConsoleChatMessage | None:
        session_id = self._message_session_index.get(message_id)
        if session_id is None:
            return None
        return self._nodes_by_session.get(session_id, {}).get(message_id)

    @staticmethod
    def _trajectory_tool_payload(content: str, tool_output_full: str | None) -> str:
        """Build the ``payload_json`` for one tool record.

        ``name`` is best-effort parsed from the marker text. File-shaped and
        hidden-reasoning outputs are omitted; other outputs pass through the
        shared credential/path scrubber and its bounded-summary limit.
        ``args`` is ``None`` because this seam does not observe arguments.
        """
        text = content or ""
        name: str | None = None
        if text.startswith("⚙ "):
            name = text[2:].split(" →", 1)[0].strip() or None
        raw_result = tool_output_full if tool_output_full is not None else text
        file_result = bool(name) and (
            name.startswith("fs_")
            or name
            in {
                "read_file",
                "write_file",
                "list_directory",
                "glob_files",
                "grep_files",
                "read_skill_file",
                "run_skill_script",
            }
        )
        contains_private_key = (
            "-----BEGIN " in raw_result.upper()
            and "PRIVATE KEY-----" in raw_result.upper()
        )
        hidden_reasoning = any(
            marker in raw_result.lower()
            for marker in ("reasoning_content", "chain of thought")
        )
        path_result = contains_local_path(raw_result)
        scrubbed = redact_log_line(raw_result)
        if file_result or hidden_reasoning or path_result:
            result = ""
            result_state = "omitted"
        elif contains_private_key:
            result = REDACTION_MARKER
            result_state = "redacted"
        else:
            result = scrubbed
            result_state = (
                "redacted"
                if REDACTION_MARKER in scrubbed
                else "truncated"
                if scrubbed != raw_result
                else "observed"
            )
        payload: dict[str, Any] = {
            "name": name,
            "args": None,
            "result": result,
            "field_states": {"args": "not_available", "result": result_state},
            "sensitivity": "path" if path_result else "tool_content",
        }
        if result_state == "truncated":
            payload["truncated"] = True
        return json.dumps(payload)

    def _record_trajectory_tool_marker(
        self,
        session_id: str,
        anchor: ConsoleChatMessage | None,
        content: str,
        tool_output_full: str | None,
    ) -> None:
        """Capture one TOOL marker's trajectory records at append time.

        TOOL-marker invariant: the marker itself is NEVER persisted to
        ``messages`` -- its ``tool_call``/``tool_result`` rows live entirely
        in the sidecar, keyed to the parent (anchor) assistant message's
        persisted id. When the anchor has no durable id yet (a marker that
        arrives while the assistant row is still streaming), the payload is
        stashed and flushed -- remapped -- when the anchor persists.
        """
        try:
            session = self._sessions.get(session_id)
            conversation_id = (
                session.persisted_conversation_id if session is not None else None
            )
            payload_json = self._trajectory_tool_payload(content, tool_output_full)
            now = time.time()
            if (
                conversation_id is not None
                and anchor is not None
                and anchor.persisted_message_id is not None
            ):
                turn_id = self._trajectory_turn_id(session_id, anchor)
                rows = self._trajectory_tool_rows(
                    message_id=anchor.persisted_message_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    payload_json=payload_json,
                    captured_at=now,
                )
                self.write_trajectory_rows(rows)
                return
            anchor_key = anchor.id if anchor is not None else "__unanchored__"
            self._pending_trajectory_tool_rows.setdefault(anchor_key, []).append(
                {
                    "session_id": session_id,
                    "payload_json": payload_json,
                    "captured_at": now,
                }
            )
        except Exception as exc:
            logger.bind(session_id=session_id, error=repr(exc)).warning(
                "trajectory_tool_marker_capture_failed"
            )

    @staticmethod
    def _trajectory_tool_rows(
        *,
        message_id: str,
        conversation_id: str,
        turn_id: str,
        payload_json: str,
        captured_at: float,
    ) -> list[TrajectoryRowWrite]:
        """Build the ``tool_call`` + ``tool_result`` row pair for one marker."""
        shared = dict(
            message_id=message_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            seq=None,
            payload_json=payload_json,
            step_started_at=captured_at,
            completed_at=captured_at,
        )
        return [
            TrajectoryRowWrite(event_kind="tool_call", **shared),
            TrajectoryRowWrite(event_kind="tool_result", **shared),
        ]

    def _trajectory_turn_id(self, session_id: str, message: ConsoleChatMessage) -> str:
        """Resolve (and memoize) the turn id for a persisted message.

        A USER message opens a turn and registers its id as the session's
        current turn; an ASSISTANT message inherits the open turn, falling
        back to its own id for assistant-first sessions.
        """
        if message.turn_id:
            return message.turn_id
        if message.role is ConsoleMessageRole.USER:
            turn_id = message.persisted_message_id or message.id
            self._session_turn_ids[session_id] = turn_id
        else:
            turn_id = self._session_turn_ids.get(session_id) or message.id
        message.turn_id = turn_id
        return turn_id

    def record_feedback_event(
        self,
        session_id: str,
        *,
        anchor_message_id: str,
        action: str,
        quote: str,
        comment: str | None = None,
    ) -> bool:
        """Persist one selection-feedback event to the trajectory sidecar.

        task-17169 (phase 4). Console selection feedback -- Request changes /
        LGTM / Comment -- was ephemeral: composed into the next user message
        and forgotten. It lands here rather than in a new annotations table
        because it is a chronological run event, and because the sidecar is
        LOCAL-ONLY (see the table's migration note): an audit record of what
        THIS device's operator said about THIS run carries no sync-schema
        implications, where a synced annotations table would.

        ``anchor_message_id`` is the NATIVE Console id carried by the quoted
        transcript row; the stored row keys off that message's persisted id,
        so an unpersisted anchor (ephemeral session, or a message not yet
        written) has nothing to anchor to and is skipped. ``seq=None`` lets
        the write transaction assign the next ledger seq, so repeated
        feedback on the same message does not collide on the
        ``(message_id, event_kind, seq)`` primary key.

        Returns True only when a row was actually written. Never raises --
        the caller is a UI dispatch path, and losing an audit record must
        never cost the user their actual feedback message.
        """
        try:
            try:
                message = self._message_or_raise(anchor_message_id)
            except KeyError:
                logger.bind(
                    session_id=session_id, anchor_message_id=anchor_message_id
                ).warning("feedback_event_unknown_anchor")
                return False
            if message.persisted_message_id is None:
                return False
            session = self._sessions.get(session_id) if session_id else None
            conversation_id = (
                session.persisted_conversation_id if session is not None else None
            )
            if conversation_id is None:
                return False
            payload: dict[str, str] = {"action": action, "quote": quote}
            # No empty-string comment: LGTM and Request-changes genuinely have
            # none, and the viewer must be able to tell "no comment" from
            # "comment the user left blank".
            if comment:
                payload["comment"] = comment
            row = TrajectoryRowWrite(
                message_id=message.persisted_message_id,
                conversation_id=conversation_id,
                turn_id=self._trajectory_turn_id(session_id, message),
                seq=None,
                event_kind="user_feedback",
                step_started_at=time.time(),
                payload_json=json.dumps(payload),
            )
            return self.write_trajectory_rows([row])
        except Exception as exc:
            logger.bind(
                session_id=session_id,
                anchor_message_id=anchor_message_id,
                error=repr(exc),
            ).warning("feedback_event_write_failed")
            return False

    def record_trace_event(
        self,
        session_id: str,
        *,
        anchor_message_id: str,
        event_kind: str,
        summary: str,
        status: str = "observed",
        event_id: str | None = None,
        parent_event_id: str | None = None,
        source_event_id: str | None = None,
        replacement_event_id: str | None = None,
        sensitivity: str = "diagnostic",
        field_states: Mapping[str, str] | None = None,
    ) -> bool:
        """Append one payload-free mutation/context observation; never raises."""
        try:
            message = self._message_or_raise(anchor_message_id)
            session = self._sessions.get(session_id)
            conversation_id = (
                session.persisted_conversation_id if session is not None else None
            )
            payload = {
                "summary": summary,
                "status": status,
                "event_id": event_id,
                "parent_event_id": parent_event_id,
                "source_event_id": source_event_id,
                "replacement_event_id": replacement_event_id,
                "field_states": {
                    "payload": "omitted",
                    **dict(field_states or {}),
                },
                "sensitivity": sensitivity,
            }
            captured_at = time.time()
            if conversation_id is None or message.persisted_message_id is None:
                self._pending_trajectory_event_rows.setdefault(message.id, []).append(
                    {
                        "event_kind": event_kind,
                        "payload_json": json.dumps(payload),
                        "captured_at": captured_at,
                    }
                )
                return True
            row = TrajectoryRowWrite(
                message_id=message.persisted_message_id,
                conversation_id=conversation_id,
                turn_id=self._trajectory_turn_id(session_id, message),
                seq=None,
                event_kind=event_kind,
                step_started_at=captured_at,
                payload_json=json.dumps(payload),
            )
            return self.write_trajectory_rows([row])
        except Exception as exc:  # noqa: BLE001 — capture is never load-bearing
            logger.warning(
                "trace_event_write_failed event_kind={} error_type={}",
                event_kind,
                type(exc).__name__,
            )
            return False

    def record_feedback_annotation(
        self,
        session_id: str,
        *,
        anchor_message_id: str,
        quote: str,
        comment: str,
    ) -> str | None:
        """Persist one Comment as a transcript annotation (task-17169 slice 2).

        The second half of the both-homes decision: alongside the
        ``user_feedback`` sidecar event (``record_feedback_event``), a
        Comment on a selected span persists as a row-anchored annotation so
        the transcript can carry an inline marker. ``row_key`` follows the
        spike's rule -- ``message:<persisted_message_id>`` -- so only
        anchors with a durable identity persist; TOOL markers, diff rows
        and ephemeral messages have none and are skipped (the spec's
        "excluded from annotation" case).

        Returns the annotation id, or ``None`` for every skip (unknown or
        unpersisted anchor, unpersisted session, no DB). Never raises: the
        caller is the same UI dispatch path as the sidecar write, and a
        lost marker must never cost the user their feedback message.
        """
        try:
            database = (
                getattr(self.persistence, "db", None) if self.persistence else None
            )
            if database is None:
                return None
            try:
                message = self._message_or_raise(anchor_message_id)
            except KeyError:
                logger.bind(
                    session_id=session_id, anchor_message_id=anchor_message_id
                ).warning("feedback_annotation_unknown_anchor")
                return None
            if message.persisted_message_id is None:
                return None
            session = self._sessions.get(session_id) if session_id else None
            conversation_id = (
                session.persisted_conversation_id if session is not None else None
            )
            if conversation_id is None:
                return None
            return database.upsert_transcript_annotation(
                conversation_id=conversation_id,
                row_key=f"message:{message.persisted_message_id}",
                message_id=message.persisted_message_id,
                quote_text=quote,
                comment=comment,
            )
        except Exception as exc:
            logger.bind(
                session_id=session_id,
                anchor_message_id=anchor_message_id,
                error=repr(exc),
            ).warning("feedback_annotation_write_failed")
            return None

    def _write_trajectory_row_for_message(self, message: ConsoleChatMessage) -> None:
        """Write one persisted message's sidecar row (and any stashed tool rows).

        The single writer for ``user``/``assistant`` rows, called from the
        persist seam (``_persist_new_message``) and the finalize flush
        (``record_trajectory_timing(flush=True)``). Idempotent per message:
        once written, later flushes are no-ops. Never raises.
        """
        try:
            # LOAD-BEARING invariant (final review): this write is TERMINAL.
            # The row snapshots ``self._trajectory_timing`` at write time and
            # the ``_trajectory_written_ids`` guard below makes every later
            # flush a no-op. ``completed_at``/``model``/``provider`` therefore
            # land ONLY if usage/timing is attached (via
            # ``record_trajectory_timing``) BEFORE the terminal persist mark.
            # A future path that persists first will silently lose those
            # facts -- there is no update, only a dropped write.
            if message.id in self._trajectory_written_ids:
                return
            if message.role not in (
                ConsoleMessageRole.USER,
                ConsoleMessageRole.ASSISTANT,
            ):
                return
            if message.persisted_message_id is None:
                return
            session_id = self._message_session_index.get(message.id)
            session = self._sessions.get(session_id) if session_id else None
            conversation_id = (
                session.persisted_conversation_id if session is not None else None
            )
            if conversation_id is None:
                return
            turn_id = self._trajectory_turn_id(session_id, message)
            timing = self._trajectory_timing.get(message.id, {})
            event_kind = (
                "user" if message.role is ConsoleMessageRole.USER else "assistant"
            )
            message_row = TrajectoryRowWrite(
                message_id=message.persisted_message_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                seq=None,
                event_kind=event_kind,
                # User records get a step-start only (spec: no token
                # boundaries on the user's own action); assistant rows
                # carry whatever the controller's capture armed --
                # NULL timing when nothing was armed (never fabricated).
                step_started_at=(
                    time.time()
                    if event_kind == "user"
                    else timing.get("step_started_at")
                ),
                first_token_at=timing.get("first_token_at"),
                completed_at=timing.get("completed_at"),
                model=timing.get("model"),
                provider=timing.get("provider"),
                payload_json=(
                    json.dumps(
                        {
                            "trace_version": 2,
                            "model_status": timing.get("model_status"),
                        }
                    )
                    if event_kind == "assistant"
                    and timing.get("step_started_at") is not None
                    else None
                ),
            )
            rows: list[TrajectoryRowWrite] = (
                [message_row] if event_kind == "user" else []
            )
            pending = self._pending_trajectory_tool_rows.pop(message.id, None)
            for entry in pending or ():
                rows.extend(
                    self._trajectory_tool_rows(
                        message_id=message.persisted_message_id,
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        payload_json=entry["payload_json"],
                        captured_at=entry["captured_at"],
                    )
                )
            pending_events = self._pending_trajectory_event_rows.pop(message.id, None)
            for entry in pending_events or ():
                rows.append(
                    TrajectoryRowWrite(
                        message_id=message.persisted_message_id,
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        seq=None,
                        event_kind=entry["event_kind"],
                        step_started_at=entry["captured_at"],
                        payload_json=entry["payload_json"],
                    )
                )
            if event_kind == "assistant":
                rows.append(message_row)
            if self.write_trajectory_rows(rows):
                self._trajectory_written_ids.add(message.id)
        except Exception as exc:
            logger.bind(message_id=message.id, error=repr(exc)).warning(
                "trajectory_row_write_failed"
            )

    def _flush_pending_trace_events_to_parent(
        self, message: ConsoleChatMessage
    ) -> None:
        """Preserve terminal observations when an empty child has no DB row."""
        pending = self._pending_trajectory_event_rows.pop(message.id, None)
        if not pending:
            return
        session_id = self._message_session_index.get(message.id)
        parent_native_id = self._native_parent_by_message.get(message.id)
        parent = self._nodes_lookup(parent_native_id) if parent_native_id else None
        session = self._sessions.get(session_id) if session_id else None
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
        if (
            parent is None
            or parent.persisted_message_id is None
            or conversation_id is None
        ):
            return
        rows = [
            TrajectoryRowWrite(
                message_id=parent.persisted_message_id,
                conversation_id=conversation_id,
                turn_id=message.turn_id or self._trajectory_turn_id(session_id, parent),
                seq=None,
                event_kind=entry["event_kind"],
                step_started_at=entry["captured_at"],
                payload_json=entry["payload_json"],
            )
            for entry in pending
        ]
        self.write_trajectory_rows(rows)

    def reset_stream_content(self, message_id: str) -> ConsoleChatMessage:
        """Discard streamed content once a turn is reclassified as a tool call.

        A disobedient model can stream prose before finally emitting a tool
        fence; the streaming adapter forwards that prose live, before the
        turn is known to be a tool call rather than a final answer. Once the
        loop classifies the completed turn as a tool call, the leaked prose
        already lives in that turn's ``STEP_MODEL`` step summary/log -- its
        rightful home -- so it is discarded here rather than left to
        concatenate onto the real final answer's chunks on the next turn
        (Plan-B Task 5 Finding A). The message is kept in the ``streaming``
        status (not reset to ``pending``) so the next turn's chunks continue
        to append normally via ``append_stream_chunk``.

        A message already ``"stopped"`` is left untouched rather than
        resurrected back to ``"streaming"`` -- mirrors ``append_stream_chunk``'s
        hardening (Plan-B agent-runtime gate Finding 1 / final-review LOW-1,
        task-227): the stop/cancel race can leave a still-running bridge
        thread calling this after the user already stopped the message, and
        that must be a benign no-op, not un-stop it.

        Args:
            message_id: Native Console message ID whose streamed content
                (buffered chunks and materialized ``content``) should be
                discarded.

        Returns:
            A snapshot of the now-empty, still-streaming message -- or the
            unmodified message, if it was already ``"stopped"``.

        Raises:
            KeyError: If the message is unknown.
            ValueError: If the message is not an assistant message.
        """
        message = self._message_or_raise(message_id)
        if message.status == "stopped":
            return self._snapshot(message)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can reset stream content.")
        message.content = ""
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        message.status = "streaming"
        self._bump_message_speech_revision(message.id)
        return self._snapshot(message)

    def set_message_usage(
        self, message_id: str, usage: ProviderUsage
    ) -> ConsoleChatMessage:
        """Attach normalized usage; persist now if the message is already terminal.

        In the normal ordering the caller attaches usage while the message is
        still ``pending``/``streaming`` and the terminal mark that follows
        flushes it. The Stop path inverts that: ``stop_active_run`` finalizes
        the message synchronously (``mark_message_stopped`` -> terminal
        flush) and only THEN cancels the stream task, whose ``CancelledError``
        handler attaches the partial usage -- against an already-terminal
        message whose second ``_mark_stream_stopped`` takes the read-back
        branch and never persists again. Without this flush, a stopped
        turn's real, already-billed input tokens were dropped on the floor
        (final-review F3).

        A stopped/failed REGENERATE is the exception: it restores the message
        to the pre-regenerate answer, so a late attach from the abandoned run
        no longer describes what the message says and is ignored (see
        ``_variant_restored_message_ids``).
        """
        message = self._message_or_raise(message_id)
        if message.id in self._variant_restored_message_ids:
            # The visible content was restored to the ORIGINAL generation's
            # answer, so this usage belongs to a run whose output no longer
            # exists here. Dropping the attach is what keeps the original's
            # own durable record intact -- combined with the flush below, a
            # late attach would otherwise overwrite it in the DB, leaving the
            # original answer priced by the abandoned regeneration.
            return self._snapshot(message)
        message.usage = usage
        if message.status not in {"pending", "streaming"}:
            self._persist_usage_only(message)
        return self._snapshot(message)

    def attach_message_exchanges(
        self, message_id: str, captures: Sequence["ExchangeCapture"]
    ) -> None:
        """Attach captured exchanges; flush now if the message is terminal.

        Mirrors ``set_message_usage``'s stop-path contract (terminal mark
        first, late attach flushes itself) with ONE deliberate divergence: a
        variant-restored message KEEPS incoming captures, marked abandoned
        (spec owner decision 6) -- the traffic really happened; usage drops
        here because it would misprice the restored answer, but captures
        carry their own ``run_tag`` and cannot misattribute.

        Dedup is by ``(run_tag, seq)`` within this merge: a capture whose
        key already exists on the message is normally dropped in favor of
        the FIRST-attached capture for that key, EXCEPT that a "stopped"
        snapshot is replaced by a later, non-"stopped" capture for the same
        key -- a stop-time partial snapshot superseded by the run's actual
        closed outcome. The DB upsert (``append_message_exchanges_local``)
        is keyed the same way, so a repeat flush of the same key is always
        harmless.
        """
        with self._capture_quiescence_lock:
            self._attach_message_exchanges_locked(message_id, captures)

    def _attach_message_exchanges_locked(
        self, message_id: str, captures: Sequence["ExchangeCapture"]
    ) -> None:
        message = self._message_or_raise(message_id)
        if self._message_session_index[message_id] in self._capture_quiescent_sessions:
            return
        abandoned = message.id in self._variant_restored_message_ids
        merged = {(c.run_tag, c.seq): c for c in message.exchanges}
        for capture in captures:
            key = (capture.run_tag, capture.seq)
            existing = merged.get(key)
            if existing is None or (
                existing.status == "stopped" and capture.status != "stopped"
            ):
                merged[key] = capture
        message.exchanges = tuple(
            sorted(merged.values(), key=lambda c: (c.run_tag, c.seq))
        )
        if abandoned:
            abandoned_tags = self._abandoned_exchange_run_tags.get(message.id)
            if not isinstance(abandoned_tags, set):
                abandoned_tags = set(abandoned_tags or ())
                self._abandoned_exchange_run_tags[message.id] = abandoned_tags
            abandoned_tags.update(c.run_tag for c in captures)
        if message.status not in {"pending", "streaming"}:
            self._persist_exchanges_only(message)

    def abandoned_exchange_run_tags(self, message_id: str) -> frozenset[str]:
        """Public read of ``_abandoned_exchange_run_tags`` for one message.

        task-9: the Conversation Inspector's Exchange tab needs to render
        the "abandoned regeneration" badge for a NATIVE (in-memory, not-yet
        -persisted) capture too, not just a DB-sourced one -- this closes
        the "known gap" ``_build_console_inspector_exchanges_loader``'s
        docstring used to describe (a native capture always reporting
        ``abandoned=False``). Returns an immutable snapshot -- the caller
        gets no handle on the private mutable set.
        """
        return frozenset(self._abandoned_exchange_run_tags.get(message_id, ()))

    def _finalize_character_emote_capture(
        self,
        message: ConsoleChatMessage,
        *,
        outcome: str,
    ) -> None:
        """Finalize sanitized content and attach local-only expression metadata."""

        capture = self._character_emote_captures.get(message.id)
        if capture is None:
            return
        successful = outcome in {"complete", "variant"}
        if successful:
            terminal = capture.parser.flush()
            if terminal.visible_text:
                self._stream_chunks_by_message.setdefault(message.id, []).append(
                    terminal.visible_text
                )
            if terminal.events and not capture.fail_closed:
                capture.events.extend(terminal.events)
                self._publish_character_emote_events(message.id, terminal.events)
        else:
            capture.parser.cancel()

        self._materialize_stream_buffer(message)
        mood_label: str | None = None
        mood_confidence: float | None = None
        mood_topic: str | None = None
        fallback_reason = capture.snapshot.fallback_reason
        if capture.events:
            mood_label = capture.events[-1].state
        elif successful and not capture.fail_closed:
            try:
                # Runs at most once per completed character turn; measured
                # ~2.3 ms at 16k chars (TASK-22227) -- bounded, kept on-loop.
                detected = detect_character_mood(
                    assistant_text=message.content,
                    user_text=self._preceding_user_text(message.id),
                )
                mood_label = detected.label
                mood_confidence = detected.confidence
                mood_topic = detected.topic
            except Exception:
                fallback_reason = "heuristic_error"
        elif outcome in {"stopped", "failed"}:
            fallback_reason = outcome

        asset = (
            capture.snapshot.asset_for_state(mood_label)
            if mood_label is not None
            else None
        )
        if capture.fail_closed:
            fallback_reason = "parser_error"
        elif (
            mood_label is not None
            and asset is None
            and fallback_reason != "resolver_error"
        ):
            fallback_reason = (
                "no_active_pack"
                if capture.snapshot.pack_version_id is None
                else "state_unavailable"
            )

        emote_metadata = CharacterEmoteMetadata(
            sanitized_utf16_length=utf16_length(message.content),
            mood_label=mood_label,
            mood_confidence=mood_confidence,
            mood_topic=mood_topic,
            emote_events=tuple(
                CharacterEmoteEventMetadata(event.state, event.at_char)
                for event in capture.events
            ),
            actor_kind=("character" if capture.snapshot.actor_id is not None else ""),
            actor_id=capture.snapshot.actor_id,
            pack_id=capture.snapshot.pack_id,
            pack_version_id=capture.snapshot.pack_version_id,
            expression_key=asset.expression_key if asset is not None else None,
            expression_id=asset.expression_id if asset is not None else None,
            asset_id=asset.asset_id if asset is not None else None,
            fallback_reason=fallback_reason,
        )
        message.metadata = replace(
            message.metadata or MessageMetadata(),
            character_emote=emote_metadata,
        )
        self._character_emote_captures.pop(message.id, None)

    def _preceding_user_text(self, message_id: str) -> str | None:
        session_id = self._message_session_index[message_id]
        message_ids = self.active_path_message_ids(session_id)
        try:
            index = message_ids.index(message_id)
        except ValueError:
            return None
        nodes = self._nodes_by_session[session_id]
        for candidate_id in reversed(message_ids[:index]):
            candidate = nodes[candidate_id]
            if candidate.role is ConsoleMessageRole.USER:
                return candidate.content
        return None

    def set_message_metadata(
        self, message_id: str, metadata: MessageMetadata
    ) -> ConsoleChatMessage:
        """Record structured facts about a turn, flushing when it is durable.

        Unlike usage -- which arrives once, at the end -- metadata is
        revised in place while a turn runs: a realtime user row is created
        ``pending`` at turn-commit and becomes ``final``/``empty``/``failed``
        when its transcript resolves, and a reply is marked ``interrupted``
        after the fact. So this always overwrites (the caller composes the
        whole record; there is no partial merge to get wrong) and persists
        immediately WHEN there is a durable row to write to.

        A row with no persisted id yet is left alone on purpose: an empty
        realtime user row is not written at all until its transcript lands
        (``_persist_new_message_or_defer``), and the create that eventually
        happens carries this metadata with it -- one write, not two.

        Args:
            message_id: Native (in-memory) message id.
            metadata: The complete metadata record for the message.

        Returns:
            An independent snapshot of the updated message.
        """
        message = self._message_or_raise(message_id)
        message.metadata = metadata
        if message.persisted_message_id is not None:
            self._persist_metadata_only(message)
        return self._snapshot(message)

    def mark_message_complete(self, message_id: str) -> ConsoleChatMessage:
        """Mark a message complete and flush final visible content to persistence."""
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
        self._finalize_character_emote_capture(message, outcome="complete")
        self._materialize_stream_buffer(message)
        retry_status = message.status
        retry_generation_state = message.assistant_generation_state
        retry_thinking = message.thinking
        session_id = self._message_session_index[message.id]
        finalizer = self._terminal_citation_finalizers.pop(message.id, None)
        terminal_persistence = (
            finalizer is not None
            or message.id in self._terminal_persistence_deferred_ids
        )
        self.clear_terminal_citation_state(message.id)
        recovery = self.dispatch_recovery_for_session(session_id)
        if (
            recovery is not None
            and recovery.assistant_message_id == message.id
            and recovery.in_flight
            # TASK-22302: do not let this shortcut swallow a turn that still
            # owes a terminal citation write -- `finalizer` is popped above, so
            # returning here discards it. Scoped to an actual finalizer, not
            # `terminal_persistence`, which is also true for a merely DEFERRED
            # turn that has no citation to seal.
            and finalizer is None
        ):
            message.status = "complete"
            message.assistant_generation_state = "complete"
            self._settle_thinking_envelope(message, "complete")
            self._bump_message_speech_revision(message.id)
            self._bump_payload_revision(session_id)
            self._settle_failed_retry_context(message, provider_visible=True)
            self._settle_owned_dispatch_terminal(message, "complete")
            self._record_message_completed(session_id, message.id)
            return self._snapshot(message)
        if not terminal_persistence:
            message.status = "complete"
            message.assistant_generation_state = "complete"
            self._settle_thinking_envelope(message, "complete")
            self._bump_message_speech_revision(message.id)
            self._bump_payload_revision(session_id)
            self._settle_failed_retry_context(message, provider_visible=True)
            try:
                self._persist_terminal_generation(message)
            except Exception:
                if message.persisted_message_id is None:
                    message.status = retry_status
                    message.assistant_generation_state = retry_generation_state
                    message.thinking = retry_thinking
                    self._pending_persistence_message_ids.add(message.id)
                raise
            self._record_message_completed(session_id, message.id)
            return self._snapshot(message)

        try:
            if not message.content:
                message.status = "complete"
                message.assistant_generation_state = "complete"
                self._settle_thinking_envelope(message, "complete")
                self._bump_message_speech_revision(message.id)
                self._bump_payload_revision(session_id)
                self._settle_failed_retry_context(message, provider_visible=True)
                self._persist_terminal_generation(message)
                # ``_persist_existing_message`` only reaches its own
                # exchanges hook when the message ALREADY had a
                # persisted_message_id -- an empty-content deferred message
                # never gets one there (no content to create a row for), so
                # this call is normally a silent no-op via
                # ``_persist_exchanges_only``'s own guard. It stays here for
                # the rare case where the row already existed.
                if message.exchanges:
                    self._persist_exchanges_only(message)
                self._record_message_completed(session_id, message.id)
                return self._snapshot(message)

            citation_write = None
            if finalizer is not None:
                try:
                    citation_write = finalizer(message.content)
                except Exception:
                    logger.warning("terminal_finalizer_unavailable")
            message.status = "complete"
            message.assistant_generation_state = "complete"
            self._settle_thinking_envelope(message, "complete")
            self._bump_message_speech_revision(message.id)
            self._bump_payload_revision(session_id)
            self._settle_failed_retry_context(message, provider_visible=True)
            try:
                if message.persisted_message_id is not None:
                    # TASK-22302: the dispatch checkpoint already wrote this row
                    # with EMPTY content, so the final body must be flushed with
                    # an UPDATE -- `create_message`'s existing-row handling lives
                    # inside its `prepared_citation is not None` branch and
                    # verifies rather than updates, so it cannot carry the body.
                    #
                    # This matters most on the FAIL-CLOSED path: `finalize()`
                    # returns None when the builder cannot seal, leaving no
                    # `citation_write` at all. Guarding this flush on one would
                    # leave the durable row empty while the in-memory message
                    # reads complete -- the answer lost, silently.
                    self._persist_terminal_generation(message)
                persisted = self._persist_new_message(
                    session_id=session_id,
                    message=message,
                    citation_write=citation_write,
                    force_stable_message_id=True,
                    terminal_persistence=True,
                )
                if not persisted:
                    if finalizer is None:
                        message.status = retry_status
                        message.assistant_generation_state = retry_generation_state
                        message.thinking = retry_thinking
                        return self._snapshot(message)
                    self._pending_persistence_message_ids.discard(message.id)
                # This branch creates the durable row directly via
                # ``_persist_new_message`` rather than
                # ``_persist_existing_message``, so it never passes through
                # that method's own exchanges hook -- flush explicitly here,
                # mirroring it, now that ``persisted_message_id`` exists.
                if message.exchanges:
                    self._persist_exchanges_only(message)
            except Exception:
                if finalizer is None and message.persisted_message_id is None:
                    message.status = retry_status
                    message.assistant_generation_state = retry_generation_state
                    message.thinking = retry_thinking
                    self._pending_persistence_message_ids.add(message.id)
                elif finalizer is not None:
                    self._pending_persistence_message_ids.discard(message.id)
                logger.warning("terminal_citation_persistence_abandoned")
                if finalizer is None:
                    return self._snapshot(message)
            self._record_message_completed(session_id, message.id)
            return self._snapshot(message)
        finally:
            self.clear_terminal_citation_state(message.id)

    def mark_message_stopped(self, message_id: str) -> ConsoleChatMessage:
        """Mark a message stopped and flush final visible content to persistence.

        If this message was mid variant-stream (regenerate), any partial
        streamed content is discarded and the pre-regenerate base content AND
        status are restored -- mirroring ``mark_message_failed`` (Plan-B
        Task 1) and the pre-refactor regenerate behavior, where Stop could
        not even reach a regenerate loop (it never set an interruptible
        task), so the original answer always survived a Stop untouched.
        Post-unification, Stop is live during regenerate; treating a stopped
        regenerate exactly like a failed one keeps that guarantee: the
        partial text is discarded (it remains recoverable from the run's own
        step log) rather than overwriting the original answer and marking it
        "stopped" (Plan-B final-review Medium-2).

        A stop with no captured base -- a normal, non-regenerate send -- has
        no known-good prior state to restore, so it keeps today's behavior:
        the partial streamed content is kept and the message is marked
        "stopped".
        """
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
        self._finalize_character_emote_capture(message, outcome="stopped")
        self._materialize_stream_buffer(message)
        session_id = self._message_session_index[message.id]
        self.clear_terminal_citation_state(message.id)
        base = self._variant_stream_bases.pop(message.id, None)
        if base is not None:
            message.content = base.content
            message.status = base.prior_status
            message.usage = base.prior_usage
            message.metadata = base.prior_metadata
            self._restore_variant_stream_base(message, base)
            self._variant_restored_message_ids.add(message.id)
        else:
            message.status = "stopped"
            message.assistant_generation_state = "stopped"
            self._settle_thinking_envelope(message, "stopped")
            self._variant_restored_message_ids.discard(message.id)
        self._bump_message_speech_revision(message.id)
        self._bump_payload_revision(session_id)
        self._settle_failed_retry_context(
            message,
            provider_visible=message.status != "failed",
        )
        if not self._settle_owned_dispatch_terminal(message, "stopped"):
            self._persist_terminal_generation(message)
            if message.persisted_message_id is None:
                self._flush_pending_trace_events_to_parent(message)
        self._settle_message_library_destination(session_id, message.id)
        return self._snapshot(message)

    def mark_message_failed(self, message_id: str) -> ConsoleChatMessage:
        """Mark a message failed and flush final visible content to persistence.

        If this message was mid variant-stream (regenerate), any partial
        streamed content is discarded and the pre-regenerate base content AND
        status are restored -- mirroring the pre-refactor regenerate
        behavior, where a failed regenerate never touched the existing
        message at all. Restoring the prior status (not just the content) is
        load-bearing: every send path builds provider context via
        ``_provider_messages_for_session(..., skip_failed=True)``, so a
        message left at "failed" status would be silently excluded from the
        model's context for the rest of the session even though its visible
        content is fully intact (Plan-B Task 1 finding).

        A failure with no captured base -- i.e. a normal, non-regenerate
        send -- has no known-good prior state to restore, so it keeps
        today's "failed" status unchanged.
        """
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
        self._finalize_character_emote_capture(message, outcome="failed")
        self._materialize_stream_buffer(message)
        session_id = self._message_session_index[message.id]
        self.clear_terminal_citation_state(message.id)
        base = self._variant_stream_bases.pop(message.id, None)
        if base is not None:
            message.content = base.content
            message.status = base.prior_status
            message.usage = base.prior_usage
            message.metadata = base.prior_metadata
            self._restore_variant_stream_base(message, base)
            self._variant_restored_message_ids.add(message.id)
        else:
            message.status = "failed"
            message.assistant_generation_state = "failed"
            self._settle_thinking_envelope(message, "failed")
            self._variant_restored_message_ids.discard(message.id)
        self._bump_message_speech_revision(message.id)
        self._bump_payload_revision(session_id)
        self._settle_failed_retry_context(
            message,
            provider_visible=message.status != "failed",
        )
        if not self._settle_owned_dispatch_terminal(message, "failed"):
            self._persist_terminal_generation(message)
            if message.persisted_message_id is None:
                self._flush_pending_trace_events_to_parent(message)
        self._settle_message_library_destination(session_id, message.id)
        return self._snapshot(message)

    def mark_message_send_blocked(self, message_id: str) -> ConsoleChatMessage:
        """Fail a never-streamed row so provider context (``skip_failed``) drops it.

        TASK-457(a): the optimistic USER echo appends the user's message BEFORE
        the provider readiness probe; if the provider is not ready the row stays
        visible in the transcript (the send is not silently dropped) but must NOT
        enter the NEXT send's provider context. Unlike ``mark_message_failed`` --
        the assistant stream state machine's terminal, which guards
        ``_validate_can_mark_terminal`` and restores a variant-regenerate base --
        this row never streamed, so it is a plain status flip to ``"failed"`` with
        no terminal guard or base handling. Callers use it only for such
        never-streamed rows (a USER echo rejected before any provider send).

        Args:
            message_id: Id of the never-streamed USER echo row to fail.

        Returns:
            A snapshot of the failed message.

        Raises:
            ValueError: If the row is not a USER echo, or is mid-stream. The
                optimistic echo is always a USER row; rejecting other roles /
                stream states stops a mistaken caller from flipping an
                assistant/system or in-flight row to ``"failed"`` and bypassing
                the assistant terminal-state guards (``mark_message_failed``).
        """
        message = self._message_or_raise(message_id)
        if message.role is not ConsoleMessageRole.USER:
            raise ValueError(
                "mark_message_send_blocked only fails a never-streamed USER echo "
                "row; assistant stream terminals use mark_message_failed."
            )
        if message.status in {"pending", "streaming"}:
            raise ValueError(
                "mark_message_send_blocked expects a never-streamed row, "
                "not one that is mid-stream."
            )
        message.status = "failed"
        self._bump_message_speech_revision(message.id)
        self._bump_payload_revision(self._message_session_index[message.id])
        self._persist_existing_message(message)
        return self._snapshot(message)

    def persist_message_if_needed(self, message_id: str) -> ConsoleChatMessage:
        """Flush a message appended with ``persist=False`` to durable storage.

        TASK-485: the cold-send optimistic echo is appended with ``persist=False``
        so a blocked/failed attempt leaves NO durable record — no orphan row and
        nothing that could re-enter the next send's provider context after a
        resume (the resume path reconstructs every row as ``"complete"``, so a
        persisted send-blocked row would silently lose its failed state). Once the
        send is confirmed to proceed, the echoed row is flushed here (creating the
        conversation via ``persist_session_if_needed``). Idempotent: a no-op
        without a persistence backend or once the row is already persisted.

        Args:
            message_id: Id of the deferred row to flush.

        Returns:
            A snapshot of the message.
        """
        message = self._message_or_raise(message_id)
        if self.persistence is None or message.persisted_message_id is not None:
            return self._snapshot(message)
        session_id = self._message_session_index[message.id]
        self._persist_new_message_or_defer(session_id=session_id, message=message)
        return self._snapshot(message)

    def prepare_message_retry(self, message_id: str) -> ConsoleChatMessage:
        """Prepare a failed assistant message to receive replacement stream content."""
        message = self._message_or_raise(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can be retried.")
        if message.status != "failed":
            raise ValueError(
                f"Only failed messages can be retried, not {message.status}."
            )
        if not message.thinking_actions_enabled:
            raise ConsoleThinkingCompatibilityError(
                "This conversation contains unreadable thinking data; "
                "upgrade before retrying it."
            )
        message.content = ""
        message.thinking = None
        message.opaque_thinking_json = None
        message.thinking_warning = None
        message.thinking_actions_enabled = True
        message.status = "pending"
        if (
            message.metadata is not None
            and message.metadata.character_emote is not None
        ):
            message.metadata = replace(message.metadata, character_emote=None)
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        # A new generation starts here -- see `begin_variant_stream`.
        self._variant_restored_message_ids.discard(message.id)
        self._failed_retry_message_ids.add(message.id)
        self._bump_message_speech_revision(message.id)
        self._bump_payload_revision(self._message_session_index[message.id])
        return self._snapshot(message)

    def add_variant(self, message_id: str, content: str) -> ConsoleChatMessage:
        """Add and select a regenerated variant for an assistant message."""
        message = self._message_or_raise(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can receive variants.")
        new_generation = ConsoleVariant(
            content=content,
            assistant_generation_state="complete",
        )
        session_id = self._message_session_index[message.id]
        previous_content = message.content
        previous_generation = self._generation_variant(message)
        on_active_path = self._message_is_on_active_path(message.id)
        durably_committed, committed_version = self._persist_generation_variant(
            message,
            new_generation,
            current=previous_generation,
            append_variant=True,
        )
        self._apply_generation_variant(message, new_generation)
        if not durably_committed:
            try:
                persisted = self._persist_existing_message(message)
            except Exception:
                self._apply_generation_variant(message, previous_generation)
                raise
            if not persisted:
                self._apply_generation_variant(message, previous_generation)
                raise RuntimeError("Variant persistence did not commit.")
        if committed_version is not None:
            message.provider_continuation_message_version = committed_version
            message.provider_continuation_remote = False
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        message.status = "complete"
        if message.variants is None:
            message.variants = ConsoleVariantSet.from_generations(
                turn_id=message.turn_id or message.id,
                generations=[
                    previous_generation,
                    new_generation,
                ],
                selected_index=1,
            )
        else:
            message.variants.variants.append(new_generation)
            message.variants.selected_index = len(message.variants.variants) - 1
        self._bump_message_speech_revision(message.id)
        provenance_cleared = self._clear_character_greeting_provenance(message)
        if not provenance_cleared:
            self._bump_payload_revision(session_id)
        if on_active_path and message.content != previous_content:
            self._bump_conversation_context_epoch(session_id)
        self._record_message_completed(session_id, message.id)
        return self._snapshot(message)

    @staticmethod
    def _generation_variant(message: ConsoleChatMessage) -> ConsoleVariant:
        """Capture every field owned by the message's selected generation."""
        return ConsoleVariant(
            content=message.content,
            thinking=message.thinking,
            opaque_thinking_json=message.opaque_thinking_json,
            thinking_warning=message.thinking_warning,
            thinking_actions_enabled=message.thinking_actions_enabled,
            usage=message.usage,
            metadata=message.metadata,
            provider_continuation=message.provider_continuation,
            provider_continuation_warning=message.provider_continuation_warning,
            provider_continuation_remote=message.provider_continuation_remote,
            provider_continuation_actions_enabled=(
                message.provider_continuation_actions_enabled
            ),
            assistant_generation_state=message.assistant_generation_state,
        )

    def _generation_owner_candidate(
        self,
        message: ConsoleChatMessage,
        *,
        current: ConsoleVariant,
        target: ConsoleVariant,
        selected_index: int | None,
        append_variant: bool,
    ) -> ConsoleChatMessage:
        """Build a detached post-projection owner without touching live state."""
        candidate = replace(message)
        if append_variant:
            if message.variants is None:
                generations = [current, target]
                candidate_index = 1
                turn_id = message.turn_id or message.id
            else:
                generations = [*message.variants.variants, target]
                candidate_index = len(generations) - 1
                turn_id = message.variants.turn_id
            candidate.variants = ConsoleVariantSet.from_generations(
                turn_id=turn_id,
                generations=generations,
                selected_index=candidate_index,
            )
        elif message.variants is not None:
            candidate_index = (
                message.variants.selected_index
                if selected_index is None
                else selected_index
            )
            candidate.variants = ConsoleVariantSet.from_generations(
                turn_id=message.variants.turn_id,
                generations=list(message.variants.variants),
                selected_index=candidate_index,
            )
        self._apply_generation_variant(candidate, target)
        candidate.status = (
            target.assistant_generation_state
            if target.assistant_generation_state in {"complete", "stopped", "failed"}
            else "complete"
        )
        return candidate

    @staticmethod
    def _generation_has_durable_evidence(variant: ConsoleVariant) -> bool:
        return any(
            (
                variant.thinking is not None,
                variant.opaque_thinking_json is not None,
                variant.usage is not None,
                variant.provider_continuation is not None,
            )
        )

    @staticmethod
    def _validate_generation_variant(
        message: ConsoleChatMessage, variant: ConsoleVariant
    ) -> None:
        if (
            not message.thinking_actions_enabled
            or not variant.thinking_actions_enabled
            or variant.opaque_thinking_json is not None
        ):
            raise ConsoleThinkingCompatibilityError(
                "This conversation contains unreadable thinking data; "
                "upgrade before changing generations."
            )
        dump_thinking_blocks_json(variant.thinking)
        dump_provider_continuation_json(variant.provider_continuation)
        if variant.usage is not None:
            variant.usage.to_json()

    def _persist_generation_variant(
        self,
        message: ConsoleChatMessage,
        variant: ConsoleVariant,
        *,
        current: ConsoleVariant | None = None,
        selected_index: int | None = None,
        append_variant: bool = False,
    ) -> tuple[bool, int | None]:
        """Persist a candidate without changing its live row owner."""
        current = current or self._generation_variant(message)
        self._validate_generation_variant(message, current)
        self._validate_generation_variant(message, variant)
        producer = self.sync_v2_chat_producer
        reconcile = getattr(producer, "reconcile_chat_message_intent", None)
        sync_configured = self.sync_v2_server_profile_id is not None
        has_durable_evidence = self._generation_has_durable_evidence(
            current
        ) or self._generation_has_durable_evidence(variant)
        persistence_db = getattr(self.persistence, "db", None)
        producer_source = getattr(producer, "source", None)
        committed_source_available = (
            callable(reconcile)
            and producer_source is persistence_db
            and callable(
                getattr(producer_source, "read_committed_chat_sync_intent", None)
            )
            and callable(getattr(producer_source, "get_message_by_id", None))
        )
        if sync_configured and has_durable_evidence and not committed_source_available:
            raise RuntimeError(
                "Whole-generation Sync projection requires committed-intent "
                "reconciliation with a committed-intent source."
            )
        candidate = self._generation_owner_candidate(
            message,
            current=current,
            target=variant,
            selected_index=selected_index,
            append_variant=append_variant,
        )
        if self.persistence is None or message.persisted_message_id is None:
            return False, None
        writer = getattr(
            self.persistence, "replace_assistant_generation_projection", None
        )
        if not callable(writer):
            if self._generation_has_durable_evidence(
                current
            ) or self._generation_has_durable_evidence(variant):
                raise RuntimeError("Generation projection persistence is unavailable.")
            if not self._persist_existing_message(candidate):
                raise RuntimeError("Generation projection persistence did not commit.")
            return True, candidate.provider_continuation_message_version
        committed_version = writer(
            message_id=message.persisted_message_id,
            content=variant.content,
            thinking_blocks_json=dump_thinking_blocks_json(variant.thinking),
            provider_continuation_json=dump_provider_continuation_json(
                variant.provider_continuation
            ),
            assistant_generation_state=variant.assistant_generation_state,
            usage_json=(variant.usage.to_json() if variant.usage is not None else None),
            expected_version=message.provider_continuation_message_version,
        )
        if type(committed_version) is not int or committed_version <= 0:
            raise RuntimeError("Selected generation persistence did not commit.")
        candidate.provider_continuation_message_version = committed_version
        candidate.provider_continuation_remote = False
        if sync_configured and committed_source_available:
            self._refresh_and_project_provider_continuation(candidate)
        else:
            self._enqueue_sync_v2_message_if_ready(candidate)
        return True, committed_version

    @staticmethod
    def _settle_thinking_envelope(
        message: ConsoleChatMessage, status: ThinkingStatus
    ) -> None:
        """Keep thinking status paired with a stopped or failed generation."""
        if message.thinking is None or not message.thinking.blocks:
            return
        current_round = max(block.round_ordinal for block in message.thinking.blocks)
        message.thinking = ThinkingEnvelope(
            tuple(
                replace(block, status=status)
                if block.round_ordinal == current_round
                else block
                for block in message.thinking.blocks
            )
        )

    def _persist_terminal_generation(self, message: ConsoleChatMessage) -> None:
        """Persist a terminal assistant as one paired generation projection."""
        if (
            message.role is ConsoleMessageRole.ASSISTANT
            and self.persist_selected_generation(message.id)
        ):
            return
        self._persist_existing_message(message, preserve_provider_continuation=True)

    @staticmethod
    def _apply_generation_variant(
        message: ConsoleChatMessage, variant: ConsoleVariant
    ) -> None:
        """Install one complete live generation on its assistant owner."""
        message.content = variant.content
        message.thinking = variant.thinking
        message.opaque_thinking_json = variant.opaque_thinking_json
        message.thinking_warning = variant.thinking_warning
        message.thinking_actions_enabled = variant.thinking_actions_enabled
        message.usage = variant.usage
        message.metadata = variant.metadata
        message.provider_continuation = variant.provider_continuation
        message.provider_continuation_warning = variant.provider_continuation_warning
        message.provider_continuation_remote = variant.provider_continuation_remote
        message.provider_continuation_actions_enabled = (
            variant.provider_continuation_actions_enabled
        )
        message.assistant_generation_state = variant.assistant_generation_state

    @staticmethod
    def _restore_variant_stream_base(
        message: ConsoleChatMessage, base: _VariantStreamBase
    ) -> None:
        message.thinking = base.prior_thinking
        message.opaque_thinking_json = base.prior_opaque_thinking_json
        message.thinking_warning = base.prior_thinking_warning
        message.thinking_actions_enabled = base.prior_thinking_actions_enabled
        message.provider_continuation = base.prior_provider_continuation
        message.provider_continuation_warning = base.prior_provider_continuation_warning
        message.provider_continuation_remote = base.prior_provider_continuation_remote
        message.provider_continuation_actions_enabled = (
            base.prior_provider_continuation_actions_enabled
        )
        message.assistant_generation_state = base.prior_assistant_generation_state

    def replace_message_thinking(
        self, message_id: str, envelope: ThinkingEnvelope | None
    ) -> ConsoleChatMessage:
        """Replace canonical thinking at the generation-owner seam."""
        message = self._message_or_raise(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can own thinking.")
        if not message.thinking_actions_enabled:
            raise ConsoleThinkingCompatibilityError(
                "This conversation contains a newer thinking format; "
                "upgrade before regenerating it."
            )
        # The dumper is the shared strict boundary; no provider-delta plumbing
        # belongs in this foundation seam.
        dump_thinking_blocks_json(envelope)
        message.thinking = envelope
        message.opaque_thinking_json = None
        message.thinking_warning = None
        return self._snapshot(message)

    def settle_message_thinking(
        self, message_id: str, envelope: ThinkingEnvelope
    ) -> ConsoleChatMessage:
        """Settle captured thinking, durably joining a detached terminal owner.

        Agent execution runs in a worker thread.  Stop can therefore commit
        the selected generation before that worker observes a typed thinking
        item which the provider had already delivered.  For that terminal
        race, project the whole selected generation again so content,
        thinking, continuation, state, and usage remain one optimistic write.
        Ordinary in-flight settlement stays process-local for the controller's
        existing terminal projection.
        """
        message = self._message_or_raise(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can own thinking.")
        if message.assistant_generation_state not in {"stopped", "failed"}:
            return self.replace_message_thinking(message_id, envelope)

        current = self._generation_variant(message)
        target = replace(
            current,
            thinking=envelope,
            opaque_thinking_json=None,
            thinking_warning=None,
            thinking_actions_enabled=True,
        )
        if target == current:
            return self._snapshot(message)
        durably_committed, committed_version = self._persist_generation_variant(
            message,
            target,
            current=current,
        )
        self._apply_generation_variant(message, target)
        if durably_committed and committed_version is not None:
            message.provider_continuation_message_version = committed_version
            message.provider_continuation_remote = False
        return self._snapshot(message)

    def begin_variant_stream(self, message_id: str) -> ConsoleChatMessage:
        """Snapshot current content as the base and reset the buffer for a new variant.

        Args:
            message_id: ID of the assistant message being regenerated.

        Returns:
            A snapshot of the message with its content cleared and status
            set to ``"streaming"``, ready to receive the new variant's
            chunks.

        Raises:
            KeyError: ``message_id`` does not reference a known message.
            ValueError: The message is not an assistant message.
        """
        message = self._message_or_raise(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can be regenerated.")
        if not message.thinking_actions_enabled:
            raise ConsoleThinkingCompatibilityError(
                "This conversation contains a newer thinking format; "
                "upgrade before regenerating it."
            )
        self._materialize_stream_buffer(message)
        self._variant_stream_bases[message.id] = _VariantStreamBase(
            content=message.content,
            prior_status=message.status,
            prior_usage=message.usage,
            prior_metadata=message.metadata,
            prior_thinking=message.thinking,
            prior_opaque_thinking_json=message.opaque_thinking_json,
            prior_thinking_warning=message.thinking_warning,
            prior_thinking_actions_enabled=message.thinking_actions_enabled,
            prior_provider_continuation=message.provider_continuation,
            prior_provider_continuation_warning=(message.provider_continuation_warning),
            prior_provider_continuation_remote=message.provider_continuation_remote,
            prior_provider_continuation_actions_enabled=(
                message.provider_continuation_actions_enabled
            ),
            prior_assistant_generation_state=message.assistant_generation_state,
        )
        # A new generation starts here, so this message's next usage attach
        # is legitimate again even if an earlier regenerate was abandoned.
        self._variant_restored_message_ids.discard(message.id)
        message.content = ""
        message.thinking = None
        message.opaque_thinking_json = None
        message.thinking_warning = None
        message.thinking_actions_enabled = True
        message.usage = None
        message.metadata = None
        message.provider_continuation = None
        message.provider_continuation_warning = None
        message.provider_continuation_remote = False
        message.provider_continuation_actions_enabled = True
        message.assistant_generation_state = "streaming"
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        message.status = "streaming"
        self._bump_message_speech_revision(message.id)
        self._bump_payload_revision(self._message_session_index[message.id])
        return self._snapshot(message)

    def finalize_variant_stream(self, message_id: str) -> ConsoleChatMessage:
        """Store the streamed buffer as a new selected variant beside the snapshot base.

        Args:
            message_id: ID of the assistant message previously passed to
                ``begin_variant_stream``.

        Returns:
            A snapshot of the message with the new variant selected as
            current and status set to ``"complete"``.

        Raises:
            KeyError: ``message_id`` does not reference a known message.
        """
        message = self._message_or_raise(message_id)
        if (
            message.status != "streaming"
            or message.id not in self._variant_stream_bases
        ):
            raise ValueError("Message has no active variant stream.")
        self._finalize_character_emote_capture(message, outcome="variant")
        self._materialize_stream_buffer(message)
        message.assistant_generation_state = "complete"
        new_generation = self._generation_variant(message)
        base_entry = self._variant_stream_bases.pop(message.id)
        base = base_entry.content
        session_id = self._message_session_index[message.id]
        on_active_path = self._message_is_on_active_path(message.id)
        if message.variants is None:
            base_generation = ConsoleVariant(
                content=base_entry.content,
                thinking=base_entry.prior_thinking,
                opaque_thinking_json=base_entry.prior_opaque_thinking_json,
                thinking_warning=base_entry.prior_thinking_warning,
                thinking_actions_enabled=base_entry.prior_thinking_actions_enabled,
                usage=base_entry.prior_usage,
                metadata=base_entry.prior_metadata,
                provider_continuation=base_entry.prior_provider_continuation,
                provider_continuation_warning=(
                    base_entry.prior_provider_continuation_warning
                ),
                provider_continuation_remote=(
                    base_entry.prior_provider_continuation_remote
                ),
                provider_continuation_actions_enabled=(
                    base_entry.prior_provider_continuation_actions_enabled
                ),
                assistant_generation_state=(
                    base_entry.prior_assistant_generation_state
                ),
            )
            message.variants = ConsoleVariantSet.from_generations(
                turn_id=message.turn_id or message.id,
                generations=[base_generation, new_generation],
                selected_index=1,
            )
        else:
            message.variants.variants.append(new_generation)
            message.variants.selected_index = len(message.variants.variants) - 1
        self._apply_generation_variant(message, message.variants.current)
        message.status = "complete"
        self._bump_message_speech_revision(message.id)
        provenance_cleared = False
        if (
            message.metadata is None
            and base_entry.prior_metadata is not None
            and base_entry.prior_metadata.template_kind == "character_greeting"
        ):
            message.metadata = MessageMetadata()
            self._bump_identity_revision(session_id)
            provenance_cleared = True
        else:
            provenance_cleared = self._clear_character_greeting_provenance(message)
        if not provenance_cleared:
            self._bump_payload_revision(session_id)
        if on_active_path and message.content != base:
            self._bump_conversation_context_epoch(session_id)
        if not self.persist_selected_generation(message.id):
            self._persist_existing_message(
                message, force_metadata_write=provenance_cleared
            )
        self._record_message_completed(session_id, message.id)
        return self._snapshot(message)

    def select_variant(
        self, message_id: str, selected_index: int
    ) -> ConsoleChatMessage:
        """Select one existing variant by index."""
        message = self._message_or_raise(message_id)
        if message.variants is None:
            raise ValueError("Message has no variants.")
        if selected_index < 0 or selected_index >= len(message.variants.variants):
            raise ValueError("selected_index must reference an existing variant")
        target = message.variants.variants[selected_index]
        session_id = self._message_session_index[message.id]
        previous_content = message.content
        previous_generation = self._generation_variant(message)
        previous_index = message.variants.selected_index
        previous_status = message.status
        on_active_path = self._message_is_on_active_path(message.id)
        durably_committed, committed_version = self._persist_generation_variant(
            message,
            target,
            current=previous_generation,
            selected_index=selected_index,
        )
        message.variants.selected_index = selected_index
        self._apply_generation_variant(message, target)
        message.status = (
            target.assistant_generation_state
            if target.assistant_generation_state in {"complete", "stopped", "failed"}
            else "complete"
        )
        if not durably_committed:
            try:
                persisted = self._persist_existing_message(message)
            except Exception:
                message.variants.selected_index = previous_index
                self._apply_generation_variant(message, previous_generation)
                message.status = previous_status
                raise
            if not persisted:
                message.variants.selected_index = previous_index
                self._apply_generation_variant(message, previous_generation)
                message.status = previous_status
                raise RuntimeError("Variant selection persistence did not commit.")
        if committed_version is not None:
            message.provider_continuation_message_version = committed_version
            message.provider_continuation_remote = False
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        self._bump_message_speech_revision(message.id)
        self._bump_payload_revision(session_id)
        if on_active_path and message.content != previous_content:
            self._bump_conversation_context_epoch(session_id)
        return self._snapshot(message)

    def persist_session_if_needed(
        self, session_id: str, *, strict_roleplay_context: bool = False
    ) -> str | None:
        """Persist a session once, returning its persisted conversation ID.

        Returns:
            The persisted conversation ID; ``None`` when no persistence
            adapter is configured, or when the session is temporary.

        Raises:
            ValueError: If ``runtime_backend`` is not exactly ``"local"`` or
                ``"server"``.
        """
        session = self._session_or_raise(session_id)
        # Temporary conversations (spec 2026-07-31) stop here, BEFORE the
        # already-persisted check and before the adapter is consulted. This
        # single early return is the entire durability mechanism: with no
        # conversation id, `persist_message_if_needed` and every other
        # conversation-scoped write returns early on its own.
        if session.ephemeral:
            return None
        if session.persisted_conversation_id is not None:
            self.retry_pending_workspace_projection(session_id)
            return session.persisted_conversation_id
        if self.persistence is None:
            return None
        if type(session.runtime_backend) is not str or session.runtime_backend not in {
            "local",
            "server",
        }:
            logger.bind(
                session_id=session_id,
                runtime_backend=_invalid_runtime_backend_diagnostic(
                    session.runtime_backend
                ),
            ).error("Cannot persist Console session with invalid runtime backend.")
            raise ValueError(
                "Cannot persist Console session: runtime_backend must be "
                "'local' or 'server'."
            )
        scope_type, persisted_workspace_id = self._persistence_scope(session)
        local_character_id = session.local_character_id()
        identity_kwargs = {
            "runtime_backend": session.runtime_backend,
            "assistant_kind": session.assistant_kind,
            "assistant_id": session.assistant_id,
            "assistant_authority_id": session.assistant_authority_id,
            "character_id": local_character_id,
            "character_name": (
                session.character_name if local_character_id is not None else None
            ),
        }
        create_conversation = self.persistence.create_conversation
        create_kwargs: dict[str, Any] = dict(
            conversation_title=session.title,
            workspace_id=persisted_workspace_id,
            scope_type=scope_type,
            system_prompt=session.settings.system_prompt
            if session.settings is not None
            else None,
            **identity_kwargs,
        )
        if session.speech_preferences != ConsoleSpeechPreferences():
            if not self._persistence_accepts_kwarg(
                create_conversation,
                "speech_preferences",
            ):
                raise RuntimeError(
                    "Persistence adapter cannot store staged reply-speech preferences."
                )
            create_kwargs["speech_preferences"] = session.speech_preferences
        staged_identity = self.stage_first_persistence(session_id)
        policy_snapshot = session.library_policy_holder.snapshot
        policy_candidate = ConsoleLibraryPolicyCandidate(
            auto_retrieve=policy_snapshot.auto_retrieve,
            assistant_access=policy_snapshot.assistant_access,
        )
        atomic_first_persist = getattr(
            self.persistence,
            "persist_console_conversation_with_policy",
            None,
        )
        if callable(atomic_first_persist):
            committed_policy = atomic_first_persist(
                conversation_id=staged_identity.conversation_id,
                policy_candidate=policy_candidate,
                conversation_kwargs=create_kwargs,
            )
            committed_identity = staged_identity
        else:
            created_id = create_conversation(**create_kwargs)
            committed_identity = ConsoleStagedConversationIdentity(
                conversation_id=created_id,
                title=staged_identity.title,
            )
            committed_policy = None
        self.publish_committed_identity(session_id, committed_identity)
        if isinstance(committed_policy, ConsoleLibraryPolicySnapshot):
            session.library_policy_holder.snapshot = committed_policy
            session.library_policy_holder.explicitly_staged = False
            if self.library_policy_coordinator is not None:
                self.library_policy_coordinator.register_holder(
                    session.id,
                    committed_identity.conversation_id,
                    session.library_policy_holder,
                )
        self._project_workspace_membership_after_commit(session)
        if (
            session.user_display_name_override is not None
            or session.character_system_template is not None
            or session.assistant_kind == "character"
        ) and not self._persist_roleplay_context(session):
            logger.warning("Failed to flush Console roleplay context on first persist.")
            if strict_roleplay_context:
                raise RuntimeError(
                    "Failed to flush Console roleplay context while promoting "
                    "a temporary session."
                )
        self._persist_project_instruction_state(session)
        pinned_prefill = (
            session.settings.pinned_prefill if session.settings is not None else None
        )
        if pinned_prefill:
            update_pinned = getattr(
                self.persistence, "update_conversation_pinned_prefill", None
            )
            if callable(update_pinned):
                try:
                    update_pinned(
                        conversation_id=session.persisted_conversation_id,
                        pinned_prefill=pinned_prefill,
                    )
                except Exception:
                    logger.bind(
                        session_id=session_id,
                        conversation_id=session.persisted_conversation_id,
                    ).exception("Failed to flush pinned prefill on first persist.")
        self._flush_context_policy_on_first_persist(session)
        # task-9: flush a session-held RAG retrieval scope (unpersisted-
        # session lifecycle, ``SessionScopeHolder``) through to durable
        # storage now that the conversation row exists. ``flush_to`` itself
        # no-ops when nothing was held, so this is safe to call
        # unconditionally. Requires the underlying ``CharactersRAGDB`` --
        # ``self.persistence`` is the ``ChatPersistenceService`` wrapper, so
        # the raw DB is reached via its ``db`` attribute, now a declared
        # (not merely probed) member of ``ConsoleChatPersistence`` (PR #747
        # review); persistence adapters without one (e.g. test fakes) still
        # simply skip the flush, matching every other durable write in this
        # method degrading gracefully when the seam it needs is absent --
        # but that skip must be OBSERVABLE (see the ``else`` branch below)
        # rather than a silent loss of the user's scope selection.
        persistence_db = getattr(self.persistence, "db", None)
        # Captured BEFORE the flush -- `flush_to` empties the holder on
        # success, so this is the only chance to learn what was actually
        # held (task-9 review finding 1; PR #747 review).
        held_scope = session.rag_scope_holder.scope
        if persistence_db is not None:
            flushed_scope = held_scope
            try:
                session.rag_scope_holder.flush_to(
                    persistence_db, session.persisted_conversation_id
                )
            except Exception:
                logger.bind(
                    session_id=session_id,
                    conversation_id=session.persisted_conversation_id,
                ).exception("Failed to flush RAG retrieval scope on first persist.")
            else:
                if flushed_scope is not None and self.on_scope_flushed is not None:
                    try:
                        self.on_scope_flushed(
                            session.persisted_conversation_id, flushed_scope
                        )
                    except Exception:
                        logger.bind(
                            session_id=session_id,
                            conversation_id=session.persisted_conversation_id,
                        ).exception(
                            "on_scope_flushed callback raised after a "
                            "successful RAG retrieval scope flush."
                        )
        elif held_scope is not None:
            # A scope WAS held but the persistence adapter exposes no raw
            # `db` seam to flush it through -- the holder is left untouched
            # (not emptied) so a later flush attempt could still succeed,
            # but the loss must not be silent: warn, naming the
            # conversation, so it is observable.
            logger.bind(
                session_id=session_id,
                conversation_id=session.persisted_conversation_id,
            ).warning(
                "Skipped RAG retrieval scope flush for conversation {} on "
                "first persist: persistence adapter exposes no `db` seam. "
                "The scope remains held in-memory only and was not "
                "written to durable storage.",
                session.persisted_conversation_id,
            )
        return session.persisted_conversation_id

    def _flush_context_policy_on_first_persist(
        self, session: ConsoleChatSession
    ) -> None:
        """Write staged non-empty policy after the conversation row exists."""
        if (
            session.persisted_conversation_id is None
            or session.context_policy_overrides.is_empty
            or self.persistence is None
        ):
            return
        writer = getattr(self.persistence, "update_conversation_context_policy", None)
        if not callable(writer):
            logger.warning(
                "Skipped staged Console context-policy flush: persistence "
                "adapter exposes no policy write seam."
            )
            return
        try:
            writer(
                conversation_id=session.persisted_conversation_id,
                overrides=session.context_policy_overrides,
            )
        except Exception:
            logger.error("Failed to flush Console context policy on first persist.")

    def _resolve_context_policy_on_resume(self, session_id: str) -> None:
        """Hydrate one persisted session's local policy without app-root state."""
        session = self._session_or_raise(session_id)
        if session.persisted_conversation_id is None or self.persistence is None:
            return
        reader = getattr(self.persistence, "get_conversation_context_policy", None)
        if not callable(reader):
            return
        try:
            result = reader(session.persisted_conversation_id)
        except Exception:
            session.context_policy_error = "context_policy_read_failed"
            logger.error("Failed to read Console context policy on resume.")
            return
        if isinstance(result, ConsoleContextPolicyOverrides):
            session.context_policy_overrides = result
            return
        overrides = getattr(result, "overrides", None)
        if isinstance(overrides, ConsoleContextPolicyOverrides):
            session.context_policy_overrides = overrides
        error = getattr(result, "error", None)
        session.context_policy_error = error if isinstance(error, str) else None

    def promote_ephemeral_session(
        self,
        session_id: str,
        *,
        contributions: Sequence[ConsoleTransactionContribution] = (),
    ) -> str | None:
        """Save a temporary conversation to durable storage, all or nothing.

        The persistence adapter must expose the atomic bundle operation. The
        Store stages the full tree, policy, scope, summary, attachments, and
        contributions without mutating live identities, then publishes only
        after that operation returns from its transaction. An instance-shadowed
        ``persist_session_if_needed`` cannot divert this path. An adapter without
        atomic promotion support is refused before identity allocation or writes.

        Args:
            session_id: Id of the temporary session to save.

        Returns:
            The new persisted conversation id, or ``None`` when the session
            was not temporary (already saved -- this is idempotent) or no
            persistence adapter is configured.

        Raises:
            RuntimeError: If atomic promotion is unavailable or an unresolved
                operation owns the session.
            Exception: Any atomic persistence failure, before live publication.
        """
        session = self._session_or_raise(session_id)
        if not session.ephemeral:
            self.retry_pending_workspace_projection(session_id)
            return None
        if self.persistence is None:
            return None
        if (
            session_id in self._unresolved_promotion_operations
            or self.dispatch_recovery_for_session(session_id) is not None
        ):
            raise RuntimeError(CONSOLE_EPHEMERAL_PROMOTION_BLOCK_COPY)

        atomic_promote = getattr(
            self.persistence,
            "promote_console_conversation_bundle",
            None,
        )
        if not callable(atomic_promote):
            raise RuntimeError("Persistence adapter cannot perform atomic promotion.")
        return self._promote_ephemeral_session_atomically(
            session,
            contributions=contributions,
        )

    def _promote_ephemeral_session_atomically(
        self,
        session: ConsoleChatSession,
        *,
        contributions: Sequence[ConsoleTransactionContribution],
    ) -> str:
        """Stage a complete temporary transcript and publish after commit only."""
        if self.persistence is None:
            raise RuntimeError("Console persistence is unavailable.")
        session_id = session.id
        messages = self._tree_nodes_parent_first(session_id)
        identity = self.stage_first_persistence(session_id)
        staged_message_ids = {message.id: str(uuid4()) for message in messages}
        prepared_messages: list[dict[str, object]] = []
        for message in messages:
            native_parent = self._native_parent_by_message.get(message.id)
            create_kwargs: dict[str, object] = {
                "sender": message.role.value,
                "content": message.content,
                "message_id": staged_message_ids[message.id],
                "parent_message_id": (
                    staged_message_ids[native_parent]
                    if native_parent is not None
                    else None
                ),
                "feedback": message.feedback,
            }
            if message.attachments:
                create_kwargs["attachments"] = [
                    {
                        "position": attachment.position,
                        "data": attachment.data,
                        "mime_type": attachment.mime_type,
                        "display_name": attachment.display_name,
                    }
                    for attachment in message.attachments
                    if attachment.data is not None
                ]
                create_kwargs["image_data"] = None
                create_kwargs["image_mime_type"] = None
            else:
                create_kwargs["image_data"] = message.image_data
                create_kwargs["image_mime_type"] = message.image_mime_type
            if message.generation_metadata:
                create_kwargs["generation_metadata"] = [
                    metadata.to_row(attachment.position)
                    for attachment, metadata in zip(
                        message.attachments,
                        message.generation_metadata,
                    )
                ]
            if message.usage is not None:
                create_kwargs["usage_json"] = message.usage.to_json()
            if message.video_metadata is not None:
                create_kwargs["metadata_json"] = message.video_metadata.to_json()
            elif message.metadata is not None and not message.metadata.is_empty:
                create_kwargs["metadata_json"] = message.metadata.to_json()
            prepared_messages.append(
                {"native_id": message.id, "create_kwargs": create_kwargs}
            )

        scope_type, workspace_id = self._persistence_scope(session)
        metadata: dict[str, object] = {}
        if session.rag_scope_holder.scope is not None:
            metadata["rag_scope"] = serialize_scope(session.rag_scope_holder.scope)
        pinned_prefill = (
            session.settings.pinned_prefill if session.settings is not None else None
        )
        if pinned_prefill:
            metadata[PINNED_PREFILL_METADATA_KEY] = pinned_prefill
        if (
            session.user_display_name_override is not None
            or session.character_system_template is not None
            or session.assistant_kind == "character"
        ):
            metadata = json.loads(
                merge_console_roleplay_context(
                    metadata,
                    ConsoleRoleplayContext(
                        user_name_override=session.user_display_name_override,
                        character_system_template=session.character_system_template,
                        character_name_snapshot=(
                            session.character_name
                            if session.assistant_kind == "character"
                            else None
                        ),
                    ),
                )
            )
        local_character_id = session.local_character_id()
        conversation_kwargs: dict[str, object] = {
            "conversation_title": identity.title,
            "workspace_id": workspace_id,
            "scope_type": scope_type,
            "system_prompt": (
                session.settings.system_prompt if session.settings is not None else None
            ),
            "runtime_backend": session.runtime_backend,
            "assistant_kind": session.assistant_kind,
            "assistant_id": session.assistant_id,
            "assistant_authority_id": session.assistant_authority_id,
            "character_id": local_character_id,
            "character_name": (
                session.character_name if local_character_id is not None else None
            ),
            "metadata": metadata or None,
            "speech_preferences": session.speech_preferences,
        }
        policy = session.library_policy_holder.snapshot
        summary, boundary_native_id = self._context_summary_by_session.get(
            session_id,
            (None, None),
        )
        committed_policy = self.persistence.promote_console_conversation_bundle(
            conversation_id=identity.conversation_id,
            policy_candidate=ConsoleLibraryPolicyCandidate(
                auto_retrieve=policy.auto_retrieve,
                assistant_access=policy.assistant_access,
            ),
            conversation_kwargs=conversation_kwargs,
            messages=prepared_messages,
            active_leaf_message_id=(
                staged_message_ids[self._active_leaf_by_session[session_id]]
                if self._active_leaf_by_session.get(session_id) is not None
                else None
            ),
            context_summary=summary,
            context_summary_boundary_message_id=(
                staged_message_ids[boundary_native_id]
                if boundary_native_id is not None
                else None
            ),
            contributions=contributions,
        )

        self.publish_committed_identity(session_id, identity)
        session.ephemeral = False
        session.library_policy_holder.snapshot = committed_policy
        session.library_policy_holder.explicitly_staged = False
        session.library_policy_holder.save_pending = False
        if self.library_policy_coordinator is not None:
            self.library_policy_coordinator.register_holder(
                session_id,
                identity.conversation_id,
                session.library_policy_holder,
            )
        self._project_workspace_membership_after_commit(session)
        for message in messages:
            message.persisted_message_id = staged_message_ids[message.id]
            native_parent = self._native_parent_by_message.get(message.id)
            message.parent_message_id = (
                staged_message_ids[native_parent] if native_parent is not None else None
            )
        held_scope = session.rag_scope_holder.scope
        session.rag_scope_holder.set(None)
        if held_scope is not None and self.on_scope_flushed is not None:
            try:
                self.on_scope_flushed(identity.conversation_id, held_scope)
            except Exception:
                logger.exception("on_scope_flushed callback failed after promotion.")
        self._persist_project_instruction_state(session)
        return identity.conversation_id

    def set_session_system_prompt(
        self,
        session_id: str,
        system_prompt: str | None,
    ) -> tuple[ConsoleChatSession, bool]:
        """Apply a system prompt to a session, persisting it if already saved.

        Updates the in-memory settings snapshot for the session and, when the
        session already owns a persisted conversation, writes the change
        through to durable storage so a later resume restores the same
        system prompt (Task 0 persistence seam: no update-conversation call
        path existed before this method). Only a blank/whitespace-only value
        is normalized to ``None`` (no system prompt); any other text is
        stored verbatim -- including leading/trailing whitespace and
        internal formatting -- so formatting-sensitive prompts survive
        unchanged.

        A persistence failure (missing conversation, version conflict, DB
        error) is caught and logged rather than raised: the in-memory
        mutation above already happened and is intentionally NOT rolled
        back, matching this store's existing convention elsewhere (e.g.
        ``update_message_content`` keeps its in-memory mutation even when
        the underlying persistence call fails) -- reverting here would just
        trade one inconsistency (durable state stale) for another (the
        in-memory session no longer reflecting what the user just applied).
        Callers get an honest ``persisted`` flag back so they can surface
        the failure instead of assuming the change was saved.

        Args:
            session_id: Native Console session ID to update.
            system_prompt: New system prompt text, or ``None``/blank to clear it.

        Returns:
            A ``(session, persisted)`` pair: the updated Console session,
            and whether the durable write (when one was attempted) actually
            succeeded. ``persisted`` is ``True`` when no durable write was
            needed (session not yet saved, or no persistence configured),
            and ``False`` when the session has no settings snapshot — the
            update was skipped entirely (task-402 honest-contract guard).
        """
        session = self._session_or_raise(session_id)
        if session.settings is None:
            # task-402: without a settings snapshot the update cannot take
            # effect in memory; writing it durably anyway would split-brain
            # the live session against the saved conversation. Report
            # honestly instead of silently claiming success.
            logger.bind(session_id=session_id).warning(
                "set_session_system_prompt skipped: session has no settings."
            )
            return session, False
        normalized = (
            system_prompt
            if isinstance(system_prompt, str) and system_prompt.strip()
            else None
        )
        if session.settings.system_prompt != normalized:
            session.has_user_work = True
        session.settings = replace(session.settings, system_prompt=normalized)
        cleared_character_template = session.character_system_template is not None
        if cleared_character_template:
            # A manual system-prompt edit revokes the trusted source rather
            # than allowing a later name refresh to overwrite user content.
            session.character_system_template = None
            self._bump_identity_revision(session_id)
        else:
            self._bump_payload_revision(session_id)
        persisted = True
        if (
            session.persisted_conversation_id is not None
            and self.persistence is not None
        ):
            if cleared_character_template and not self._persist_roleplay_context(
                session
            ):
                persisted = False
            update_system_prompt = getattr(
                self.persistence,
                "update_conversation_system_prompt",
                None,
            )
            if callable(update_system_prompt):
                try:
                    if not update_system_prompt(
                        conversation_id=session.persisted_conversation_id,
                        system_prompt=normalized,
                    ):
                        persisted = False
                except Exception as exc:
                    persisted = False
                    logger.error(
                        "Console operation failed "
                        "(operation=set_session_system_prompt, "
                        "context=durable_write, exception_category={}); "
                        "in-memory session keeps the applied value.",
                        type(exc).__name__,
                    )
        return session, persisted

    def set_session_pinned_prefill(
        self, session_id: str, prefill: str | None
    ) -> tuple[ConsoleChatSession, bool]:
        """Set or clear a session's pinned response prefill.

        Mirrors ``set_session_system_prompt``: updates the in-memory
        settings snapshot and, when the session already owns a persisted
        conversation, writes through to conversation metadata. A durable
        write failure is caught and logged; the in-memory value is kept and
        the honest ``persisted`` flag is returned. A session with no
        settings snapshot skips the update entirely and returns ``False``
        (task-402 honest-contract guard).

        Args:
            session_id: Native Console session ID to update.
            prefill: New pinned prefill text, or ``None``/blank to clear it.

        Returns:
            A ``(session, persisted)`` pair: the updated Console session,
            and whether the requested state fully took effect — ``False``
            when the durable write failed or the session has no settings
            snapshot; ``True`` otherwise (including when no durable write
            was needed).
        """
        session = self._session_or_raise(session_id)
        if session.settings is None:
            # task-402: mirror set_session_system_prompt -- no settings
            # snapshot means the update cannot apply in memory; skip the
            # durable write and report honestly.
            logger.bind(session_id=session_id).warning(
                "set_session_pinned_prefill skipped: session has no settings."
            )
            return session, False
        normalized = prefill if isinstance(prefill, str) and prefill.strip() else None
        if session.settings.pinned_prefill != normalized:
            session.has_user_work = True
        session.settings = replace(session.settings, pinned_prefill=normalized)
        self._bump_payload_revision(session_id)
        persisted = True
        if (
            session.persisted_conversation_id is not None
            and self.persistence is not None
        ):
            update_pinned = getattr(
                self.persistence, "update_conversation_pinned_prefill", None
            )
            if callable(update_pinned):
                try:
                    update_pinned(
                        conversation_id=session.persisted_conversation_id,
                        pinned_prefill=normalized,
                    )
                except Exception:
                    persisted = False
                    logger.bind(
                        session_id=session_id,
                        conversation_id=session.persisted_conversation_id,
                    ).exception(
                        "Failed to persist Console pinned prefill; "
                        "in-memory session keeps the applied value."
                    )
        return session, persisted

    def _persist_new_message_or_defer(
        self, *, session_id: str, message: ConsoleChatMessage
    ) -> None:
        if self.persistence is None:
            return
        if message.id in self._terminal_persistence_deferred_ids:
            self._pending_persistence_message_ids.add(message.id)
            self.persist_session_if_needed(session_id)
            return
        if not message.content and not message.attachments:
            self._pending_persistence_message_ids.add(message.id)
            self.persist_session_if_needed(session_id)
            return
        self._persist_new_message(session_id=session_id, message=message)

    def _citation_persistence_ready(self) -> bool:
        """Return whether terminal citation writes can be accepted safely."""
        persistence = self.persistence
        if persistence is None:
            return False
        try:
            parameters = inspect.signature(persistence.create_message).parameters
            accepts_citation_write = "citation_write" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            return (
                accepts_citation_write
                and getattr(
                    persistence,
                    "canonical_citation_writes_ready",
                    False,
                )
                is True
            )
        except Exception:
            return False

    def clear_terminal_citation_state(self, message_id: str) -> None:
        """Clear terminal selection and stream buffers without persisting."""
        session_id = self._message_session_index.get(message_id)
        message = self._nodes_by_session.get(session_id or "", {}).get(message_id)
        if message is not None:
            self._fold_stream_buffer_without_persistence(message)
        self._terminal_citation_finalizers.pop(message_id, None)
        self._provisional_terminal_selection_ids.discard(message_id)
        self._terminal_persistence_deferred_ids.discard(message_id)
        self._stream_chunks_by_message.pop(message_id, None)
        self._stream_materialized_counts.pop(message_id, None)

    @staticmethod
    def _persistence_accepts_kwarg(func: Any, name: str) -> bool:
        """Return True when ``func`` can be called with keyword ``name``.

        The ``attachments`` parameter was added to
        :class:`ConsoleChatPersistence` after several persistence fakes were
        already written in tests; those fakes are entitled to omit it (see
        the Protocol docstrings above). Probing the declared signature lets
        the two persist methods below pass ``attachments`` only to
        implementations that actually declare it (or accept ``**kwargs``),
        instead of raising ``TypeError`` against older/narrower fakes.
        """
        try:
            parameters = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return True
        if name in parameters:
            return True
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    def _nearest_persisted_ancestor_id(
        self, session_id: str, message: ConsoleChatMessage
    ) -> str | None:
        """Return the persisted id of ``message``'s nearest PERSISTED ancestor.

        Walks the native parent chain upward from ``message`` (via
        ``_native_parent_by_message``), skipping any ancestor that is not
        itself durably persisted (``persisted_message_id is None`` -- e.g. a
        ``persist=False`` interstitial system note the controller appended
        mid-chain), and returns the first persisted ancestor's persisted id.
        Returns ``None`` when no ancestor is persisted (the message is a true
        persisted root).

        This keeps the persisted tree connected across non-persisted tree
        nodes: without it, a message whose IMMEDIATE tree parent is a
        non-persisted node would be written with ``parent_message_id=None``
        and become a stray DB root, fragmenting the chain Task 8's leaf->root
        resume walk depends on. For a plain linear conversation with no
        interstitials the immediate parent IS the nearest persisted ancestor,
        so the resolved id is unchanged.

        A visited-set guards against a malformed cyclic parent chain.
        """
        nodes = self._nodes_by_session.get(session_id, {})
        visited: set[str] = {message.id}
        current = self._native_parent_by_message.get(message.id)
        while current is not None and current not in visited:
            visited.add(current)
            ancestor = nodes.get(current)
            if ancestor is not None and ancestor.persisted_message_id is not None:
                return ancestor.persisted_message_id
            current = self._native_parent_by_message.get(current)
        return None

    def durable_parent_for_message(self, message_id: str) -> str | None:
        """Resolve the persisted parent one durable write should thread.

        The dispatch checkpoint path used to read ``message.parent_message_id``
        directly, but that field is the *persisted* parent id and is assigned
        only by :meth:`_persist_new_message`. A ``persist=False`` optimistic
        echo has not been through that method, so the field is always ``None``
        and every checkpointed turn was written as a fresh DB root -- forking
        the conversation away from its own history.

        Args:
            message_id: Native id of the message about to be persisted.

        Returns:
            The nearest PERSISTED ancestor's persisted id -- non-persisted
            mid-chain nodes are skipped, matching :meth:`_persist_new_message`
            -- or ``None`` when nothing above it is durable, which is the
            documented "true persisted root" answer rather than an error.
        """

        session_id = self._message_session_index.get(message_id)
        if session_id is None:
            return None
        message = self._nodes_by_session.get(session_id, {}).get(message_id)
        if message is None:
            return None
        return self._nearest_persisted_ancestor_id(session_id, message)

    def _persist_new_message(
        self,
        *,
        session_id: str,
        message: ConsoleChatMessage,
        citation_write: SealedCitationWrite | None = None,
        force_stable_message_id: bool = False,
        terminal_persistence: bool = False,
    ) -> bool:
        if self.persistence is None:
            return False
        conversation_id = self.persist_session_if_needed(session_id)
        if conversation_id is None:
            return False
        # Thread the real tree parent through to persistence, resolving to the
        # nearest PERSISTED ancestor (skipping non-persisted mid-chain nodes
        # such as ``persist=False`` interstitial notes) so the persisted tree
        # stays connected. ``None`` only when no ancestor is persisted (a true
        # persisted root) -- never a dangling id.
        parent_persisted_id = self._nearest_persisted_ancestor_id(session_id, message)
        message.parent_message_id = parent_persisted_id
        create_kwargs: dict[str, Any] = dict(
            conversation_id=conversation_id,
            sender=message.role.value,
            content=message.content,
            # Image/video generation messages pin the DB row to the SAME id as
            # the store's own native tree-node id: ``message.id`` is already
            # a globally-unique uuid4, and ``add_message`` accepts an
            # explicit id. This makes ``persisted_message_id == message.id``
            # for generation messages specifically: image variant ops can
            # address the durable row directly, and video files saved under
            # the preallocated native id remain resolvable after reload.
            # Every other message kind keeps letting the DB assign its own id.
            message_id=message.id
            if (
                message.generation_metadata
                or message.video_metadata is not None
                or force_stable_message_id
            )
            else None,
            parent_message_id=parent_persisted_id,
            feedback=message.feedback,
        )
        # Only engage split addressing when there is something beyond the
        # legacy position-0 slot to address -- a lone image (whether staged
        # via scalar kwargs or a single-item attachments tuple) keeps using
        # the scalar image_data/image_mime_type kwargs exactly as before.
        # A generation message ALWAYS engages split addressing, even with a
        # single variant, since it needs the ``message_attachments``
        # position-0 semantics narrow keep/append-variant ops rely on.
        attachments_payload = None
        if len(message.attachments) > 1 or message.generation_metadata:
            attachments_payload = [
                {
                    "position": attachment.position,
                    "data": attachment.data,
                    "mime_type": attachment.mime_type,
                    "display_name": attachment.display_name,
                }
                for attachment in message.attachments
                if attachment.data is not None
            ]
        if attachments_payload and self._persistence_accepts_kwarg(
            self.persistence.create_message, "attachments"
        ):
            create_kwargs["attachments"] = attachments_payload
            # The real service derives the legacy image_data/image_mime_type
            # columns from position 0 of ``attachments`` (overriding whatever
            # is passed here), but the kwargs are keyword-only with no
            # defaults, so they must still be supplied explicitly.
            create_kwargs["image_data"] = None
            create_kwargs["image_mime_type"] = None
        else:
            create_kwargs["image_data"] = message.image_data
            create_kwargs["image_mime_type"] = message.image_mime_type
        # Generation-metadata sidecar rows ride the SAME create_message call
        # as the attachments write above -- one atomic transaction on the
        # real service (Task 2) -- rather than a follow-up write, so a
        # sidecar failure rolls back the whole message instead of leaving an
        # image-bearing message with no generation metadata.
        if message.generation_metadata and self._persistence_accepts_kwarg(
            self.persistence.create_message, "generation_metadata"
        ):
            for attachment, meta in zip(
                message.attachments, message.generation_metadata
            ):
                if attachment.data is None:
                    raise ValueError(
                        f"generation variant at position {attachment.position} has no bytes; "
                        "generation creation always supplies fresh bytes (caller bug)."
                    )
            create_kwargs["generation_metadata"] = [
                meta.to_row(attachment.position)
                for attachment, meta in zip(
                    message.attachments, message.generation_metadata
                )
            ]
        if message.role is ConsoleMessageRole.ASSISTANT:
            generation = self._generation_variant(message)
            generation_fields = (
                "thinking_blocks_json",
                "provider_continuation_json",
                "assistant_generation_state",
                "usage_json",
            )
            accepts_generation = all(
                self._persistence_accepts_kwarg(
                    self.persistence.create_message, field_name
                )
                for field_name in generation_fields
            )
            if accepts_generation:
                self._validate_generation_variant(message, generation)
                create_kwargs.update(
                    thinking_blocks_json=dump_thinking_blocks_json(generation.thinking),
                    provider_continuation_json=dump_provider_continuation_json(
                        generation.provider_continuation
                    ),
                    assistant_generation_state=(generation.assistant_generation_state),
                    usage_json=(
                        generation.usage.to_json()
                        if generation.usage is not None
                        else None
                    ),
                )
            elif self._generation_has_durable_evidence(generation):
                raise RuntimeError(
                    "Persistence cannot atomically create this assistant generation."
                )
        elif message.usage is not None and self._persistence_accepts_kwarg(
            self.persistence.create_message, "usage_json"
        ):
            create_kwargs["usage_json"] = message.usage.to_json()
        # Video generation metadata (task-3401.4): persisted under the same
        # local-only metadata_json column as a namespaced payload. Mutually
        # exclusive with the provenance branch below by construction -- when
        # both are somehow set, the video payload wins because it is the
        # load-bearing one for the tombstone card.
        if message.video_metadata is not None and self._persistence_accepts_kwarg(
            self.persistence.create_message, "metadata_json"
        ):
            create_kwargs["metadata_json"] = message.video_metadata.to_json()
        # Structured message metadata (task-2364): same declare-to-receive
        # rule, and an all-default instance is treated as "nothing to
        # record" rather than written as a row of noise.
        elif (
            message.metadata is not None
            and not message.metadata.is_empty
            and self._persistence_accepts_kwarg(
                self.persistence.create_message, "metadata_json"
            )
        ):
            create_kwargs["metadata_json"] = message.metadata.to_json()
        if citation_write is not None:
            create_kwargs["citation_write"] = citation_write
        if terminal_persistence:
            persisted_message_id = self._create_terminal_message(
                create_kwargs=create_kwargs,
                citation_write=citation_write,
            )
            if persisted_message_id is None:
                self._pending_persistence_message_ids.add(message.id)
                return False
        else:
            persisted_message_id = self.persistence.create_message(**create_kwargs)
        message.persisted_message_id = persisted_message_id
        self._pending_persistence_message_ids.discard(message.id)
        self._refresh_generation_owner_version(message)
        # Carried-forward (Task 8): when this newly persisted message IS the
        # session's active leaf, write the durable active-leaf pointer through
        # NOW that it owns a persisted id. ``append_message`` advances the
        # in-memory leaf but (unlike ``set_active_leaf``/``create_sibling``)
        # never writes the DB pointer; without this, sending a new message on a
        # swiped-back branch leaves the pointer at the pre-swipe leaf, so a
        # later resume walks the wrong branch and drops the continuation. Also
        # covers the deferred path (``_persist_pending_message_if_ready`` ->
        # here) where the id only exists once streamed content arrives.
        if terminal_persistence:
            try:
                if message.id == self._active_leaf_by_session.get(session_id):
                    self._persist_active_leaf(
                        session_id,
                        message.id,
                        content_safe_diagnostic=True,
                    )
                self._enqueue_sync_v2_message_if_ready(
                    message,
                    content_safe_diagnostic=True,
                )
            except Exception:
                logger.warning("terminal_persistence_bookkeeping_unavailable")
        else:
            if message.id == self._active_leaf_by_session.get(session_id):
                self._persist_active_leaf(session_id, message.id)
            self._enqueue_sync_v2_message_if_ready(message)
        # Trajectory sidecar (schema v38): every persisted Console message
        # gets a user/assistant row in the LOCAL-ONLY sidecar, batched with
        # any tool rows stashed while this row was still streaming.
        # Best-effort -- never fails the persist that triggered it.
        self._write_trajectory_row_for_message(message)
        return True

    def _create_terminal_message(
        self,
        *,
        create_kwargs: dict[str, Any],
        citation_write: SealedCitationWrite | None,
    ) -> str | None:
        """Perform the bounded terminal create/fallback disposition."""
        if self.persistence is None:
            return None
        if citation_write is None:
            try:
                return self.persistence.create_message(**create_kwargs)
            except Exception:
                logger.warning("terminal_ordinary_persistence_abandoned")
                return None

        try:
            return self.persistence.create_message(**create_kwargs)
        except CitationPersistenceUnavailable:
            logger.warning("terminal_citation_persistence_fallback")
            create_kwargs.pop("citation_write", None)
            try:
                return self.persistence.create_message(**create_kwargs)
            except Exception:
                logger.warning("terminal_citation_persistence_abandoned")
                return None
        except Exception:
            logger.warning("terminal_citation_persistence_ambiguous_retry")
            try:
                return self.persistence.create_message(**create_kwargs)
            except Exception:
                logger.warning("terminal_citation_persistence_abandoned")
                return None

    def _persist_usage_only(self, message: ConsoleChatMessage) -> None:
        """Flush an already-terminal message's usage without a version bump.

        This is the Stop-path terminal flush described on
        ``set_message_usage``: the message's content/version were already
        persisted by an earlier terminal mark, and only ``usage_json`` is
        new here. Routing that through the ordinary content path
        (``_persist_existing_message`` -> ``update_message_content`` ->
        ``CharactersRAGDB.update_message``) would still bump ``version``/
        ``last_modified`` on a write where content did not change, which
        trips the ``messages_sync_update`` trigger's ``WHEN`` clause (it
        watches those two columns, not just content) and enqueues a
        ``sync_log`` row whose payload can never carry ``usage_json`` --
        cross-device churn, and a spurious optimistic-lock version bump, for
        a column that is local-only by design (Qodo round).

        Prefers the persistence adapter's ``update_message_usage`` method
        (a version-neutral, local-only column write -- see
        ``ChatPersistenceService.update_message_usage``) when the adapter
        provides one, probed the same hasattr+callable way as
        ``_persistence_accepts_kwarg`` so narrow test fakes that predate
        this method are not broken. Those older fakes -- and the
        not-yet-durably-created case -- fall back to the pre-existing
        content-carrying ``_persist_existing_message`` path unchanged, so
        behavior degrades gracefully rather than breaking.
        """
        if self.persistence is None:
            return
        if message.persisted_message_id is None or message.usage is None:
            self._persist_existing_message(message)
            return
        usage_writer = getattr(self.persistence, "update_message_usage", None)
        if callable(usage_writer):
            usage_writer(
                message_id=message.persisted_message_id,
                usage_json=message.usage.to_json(),
            )
            # Sync v2 (a separate, content-carrying sync pipeline) never
            # transmits usage_json either, and the terminal mark that
            # preceded this usage-only flush already enqueued this
            # message's content once -- re-enqueueing identical content
            # here would be the same flavor of profitless churn this fix
            # is removing from the legacy sync_log trigger path.
            return
        self._persist_existing_message(message)

    def _persist_exchanges_only(self, message: ConsoleChatMessage) -> None:
        """Flush captured provider exchanges without a version bump.

        The exchanges twin of ``_persist_usage_only``: the
        ``message_exchanges`` table carries no sync trigger at all (see the
        v40->v41 migration), so this never rides the general-purpose
        content path. Unlike usage, there is no content-carrying fallback
        to degrade to -- captures only ever reach the DB through this
        dedicated path, so a caller with no adapter, no persisted row yet,
        or nothing to write bails out silently rather than falling back to
        ``_persist_existing_message``.
        """
        with self._capture_quiescence_lock:
            self._persist_exchanges_only_locked(message)

    def _persist_exchanges_only_locked(self, message: ConsoleChatMessage) -> None:
        session_id = self._message_session_index.get(message.id)
        if session_id in self._capture_quiescent_sessions:
            return
        if self.persistence is None:
            return
        if message.persisted_message_id is None or not message.exchanges:
            return
        writer = getattr(self.persistence, "append_message_exchanges", None)
        if not callable(writer):
            return
        # Row-building (including ``capture_to_blob``'s JSON serialization)
        # lives INSIDE the try, not just the writer call: a malformed
        # capture (non-str-keyed nested dict, a circular reference in
        # ``request``/``response``) can raise during serialization, and the
        # never-raise contract here covers the whole flush, not just the
        # network/DB half of it -- a serialization error must degrade to
        # the same warning log as a write failure, never unwind past this
        # method into an already-committed terminal mark.
        try:
            abandoned_tags = self._abandoned_exchange_run_tags.get(message.id, set())
            # Qodo PR #1883 finding 4: this method runs on EVERY flush of a
            # message with exchanges (e.g. once per tool call in a long
            # agent turn), but a capture's compressed blob only needs
            # computing once per (run_tag, seq, status) -- captures are
            # frozen and, per ``attach_message_exchanges``'s merge rule,
            # the ONLY legitimate content change for an existing key is a
            # "stopped" snapshot superseded by a later non-"stopped"
            # capture, which is a STATUS change and so is naturally a cache
            # miss (a different key) rather than a stale hit.
            blob_cache = self._exchange_blob_cache.get(message.id)
            if not isinstance(blob_cache, dict):
                blob_cache = dict(blob_cache or ())
                self._exchange_blob_cache[message.id] = blob_cache
            current_keys: set[tuple[str, int, str]] = set()
            rows = []
            for c in message.exchanges:
                cache_key = (c.run_tag, c.seq, c.status)
                current_keys.add(cache_key)
                blob = blob_cache.get(cache_key)
                if blob is None:
                    blob = capture_to_blob(c)
                    blob_cache[cache_key] = blob
                rows.append(
                    {
                        "run_tag": c.run_tag,
                        "seq": c.seq,
                        "status": c.status,
                        "abandoned": c.run_tag in abandoned_tags,
                        "capture_detail": c.capture_detail.value,
                        "capture_blob": blob,
                        "created_at": c.created_at,
                    }
                )
            # Prune keys no longer on the message (e.g. a superseded
            # "stopped" blob) so this message's cache entry cannot outgrow
            # its current capture count.
            for stale_key in blob_cache.keys() - current_keys:
                del blob_cache[stale_key]
            writer(message_id=message.persisted_message_id, rows=rows)
        except Exception as exc:
            logger.bind(
                message_id=message.id,
                error_type=type(exc).__name__,
            ).warning(
                "exchange_flush_failed"
            )

    def _persist_metadata_only(self, message: ConsoleChatMessage) -> None:
        """Flush a persisted message's metadata without a version bump.

        The metadata twin of ``_persist_usage_only``, and local-only for
        the same reason: ``metadata_json`` never rides a sync payload, so
        routing this through the content path would bump
        ``version``/``last_modified`` on a write whose content did not
        change, trip the ``messages_sync_update`` trigger's ``WHEN`` clause
        and enqueue a ``sync_log`` row that cannot carry the column that
        changed.

        Prefers the adapter's ``update_message_metadata`` (probed the same
        hasattr+callable way as its usage sibling) and falls back to the
        content-carrying path for narrow fakes that predate it.
        """
        if self.persistence is None:
            return
        if message.persisted_message_id is None or (
            message.metadata is None and message.video_metadata is None
        ):
            self._persist_existing_message(message)
            return
        metadata_writer = getattr(self.persistence, "update_message_metadata", None)
        if callable(metadata_writer):
            # task-3401.4: a video row's payload lives in video_metadata and
            # is preferred, so a metadata flush can never overwrite the
            # video facts with an (all-defaults) provenance payload.
            payload = (
                message.video_metadata.to_json()
                if message.video_metadata is not None
                else message.metadata.to_json()
            )
            metadata_writer(
                message_id=message.persisted_message_id,
                metadata_json=payload,
            )
            # Sync v2 never transmits metadata_json either, and this row's
            # content was already enqueued by whichever write persisted it
            # -- re-enqueueing identical content here would be the same
            # profitless churn the local-only write exists to avoid.
            return
        self._persist_existing_message(message)

    def persist_selected_generation(self, message_id: str) -> bool:
        """Atomically project the complete selected assistant generation."""
        message = self._message_or_raise(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages own generation projections.")
        current = self._generation_variant(message)
        durably_committed, committed_version = self._persist_generation_variant(
            message, current, current=current
        )
        if not durably_committed:
            return False
        if committed_version is not None:
            message.provider_continuation_message_version = committed_version
        message.provider_continuation_remote = False
        return True

    def _persist_existing_message(
        self,
        message: ConsoleChatMessage,
        *,
        update_feedback: bool = False,
        force_metadata_write: bool = False,
        preserve_provider_continuation: bool = False,
        clear_generation_provenance: bool = False,
    ) -> bool:
        if self.persistence is None:
            return True
        if message.persisted_message_id is None:
            self._persist_pending_message_if_ready(message)
            return True
        update_kwargs: dict[str, Any] = dict(
            message_id=message.persisted_message_id,
            content=message.content,
            image_data=message.image_data,
            image_mime_type=message.image_mime_type,
            parent_message_id=None,
            feedback=message.feedback,
            update_parent=False,
            update_feedback=update_feedback,
        )
        # Edits never change attachments -- the scalar image kwargs above
        # continue to carry the #0 mirror (pre-existing preserve semantics).
        # attachments=None is sent whenever the implementation supports the
        # kwarg, telling split-addressed backends to leave the attachments
        # table alone.
        if self._persistence_accepts_kwarg(
            self.persistence.update_message_content, "attachments"
        ):
            update_kwargs["attachments"] = None
        # Normalized provider usage (Console cost ticker): forwarded only
        # when this message actually carries usage AND the adapter declares
        # the kwarg. Omitted entirely (never sent as ``None``) so a
        # content-only update (e.g. a mid-stream edit before usage is known)
        # never overwrites an already-persisted usage value with NULL.
        if message.usage is not None and self._persistence_accepts_kwarg(
            self.persistence.update_message_content, "usage_json"
        ):
            update_kwargs["usage_json"] = message.usage.to_json()
        # Structured message metadata (task-2364): omitted entirely, never
        # sent as ``None``, so a content-only update cannot NULL a value
        # already on the row. task-3401.4: a video row's payload lives in
        # ``video_metadata`` and is preferred -- a content edit rewrites the
        # same video payload (idempotent) instead of clobbering it with an
        # all-defaults provenance payload.
        if message.video_metadata is not None and self._persistence_accepts_kwarg(
            self.persistence.update_message_content, "metadata_json"
        ):
            update_kwargs["metadata_json"] = message.video_metadata.to_json()
        elif (
            message.metadata is not None
            and (force_metadata_write or not message.metadata.is_empty)
            and self._persistence_accepts_kwarg(
                self.persistence.update_message_content, "metadata_json"
            )
        ):
            update_kwargs["metadata_json"] = message.metadata.to_json()
        if self._persistence_accepts_kwarg(
            self.persistence.update_message_content,
            "preserve_provider_continuation",
        ):
            update_kwargs["preserve_provider_continuation"] = (
                preserve_provider_continuation
            )
        if clear_generation_provenance and self._persistence_accepts_kwarg(
            self.persistence.update_message_content,
            "clear_generation_provenance",
        ):
            update_kwargs["clear_generation_provenance"] = True
        had_provider_continuation = message.provider_continuation is not None
        if not self.persistence.update_message_content(**update_kwargs):
            return False
        if message.exchanges:
            self._persist_exchanges_only(message)
        refresh_preserved_continuation = had_provider_continuation
        if clear_generation_provenance and not refresh_preserved_continuation:
            database = getattr(self.persistence, "db", None)
            getter = getattr(database, "get_message_by_id", None)
            if callable(getter):
                committed = getter(message.persisted_message_id)
                refresh_preserved_continuation = bool(
                    committed is not None
                    and committed.get("provider_continuation_json") is not None
                )
        if refresh_preserved_continuation:
            self._refresh_and_project_provider_continuation(message)
            return True
        self._refresh_generation_owner_version(message)
        self._enqueue_sync_v2_message_if_ready(message)
        return True

    def _refresh_generation_owner_version(self, message: ConsoleChatMessage) -> None:
        """Refresh the committed version when the adapter exposes its local DB."""
        persisted_id = message.persisted_message_id
        database = getattr(self.persistence, "db", None) if self.persistence else None
        getter = getattr(database, "get_message_by_id", None)
        if persisted_id is None or not callable(getter):
            return
        row = getter(persisted_id)
        if row is not None and type(row.get("version")) is int:
            message.provider_continuation_message_version = row["version"]

    def _refresh_and_project_provider_continuation(
        self, message: ConsoleChatMessage
    ) -> None:
        """Refresh one continuation owner and project its exact committed row."""
        persisted_id = message.persisted_message_id
        database = getattr(self.persistence, "db", None) if self.persistence else None
        getter = getattr(database, "get_message_by_id", None)
        if persisted_id is None or not callable(getter):
            raise RuntimeError("Durable continuation owner is unavailable.")
        row = getter(persisted_id)
        if row is None:
            raise RuntimeError("Durable continuation owner is unavailable.")
        safe = read_provider_continuation_json(row.get("provider_continuation_json"))
        thinking = read_thinking_blocks_json(row.get("thinking_blocks_json"))
        if thinking.warning is not None or thinking.opaque_json is not None:
            raise RuntimeError("Durable thinking owner is unreadable.")
        message.provider_continuation = safe.checkpoint
        message.provider_continuation_message_version = int(row["version"])
        message.provider_continuation_remote = False
        message.provider_continuation_warning = safe.warning
        message.assistant_generation_state = row.get("assistant_generation_state")
        message.provider_continuation_actions_enabled = safe.checkpoint is not None

        producer = self.sync_v2_chat_producer
        profile_id = self.sync_v2_server_profile_id
        reconcile = getattr(producer, "reconcile_chat_message_intent", None)
        if profile_id is None or not callable(reconcile):
            return
        payload: dict[str, Any] = {
            "assistant_generation_state": row.get("assistant_generation_state"),
            "content": str(row.get("content") or ""),
            "role": message.role.value,
        }
        if safe.checkpoint is not None:
            payload["provider_continuation_json"] = dump_provider_continuation_json(
                safe.checkpoint
            )
        if thinking.envelope is not None:
            payload["thinking_blocks_json"] = dump_thinking_blocks_json(
                thinking.envelope
            )
        try:
            reconcile(
                server_profile_id=profile_id,
                authenticated_principal_id=self.sync_v2_authenticated_principal_id,
                workspace_scope=self.sync_v2_workspace_scope,
                message_id=persisted_id,
                message_version=int(row["version"]),
                payload_hash=canonical_payload_hash(payload),
            )
        except Exception:
            logger.warning(
                "Failed to project Sync v2 continuation owner after local mutation"
            )

    def _persist_pending_message_if_ready(self, message: ConsoleChatMessage) -> None:
        if (
            self.persistence is None
            or message.id not in self._pending_persistence_message_ids
            or message.id in self._terminal_persistence_deferred_ids
            or not message.content
        ):
            return
        session_id = self._message_session_index[message.id]
        self._persist_new_message(session_id=session_id, message=message)

    def ensure_provider_continuation_durable(
        self,
        *,
        message_id: str,
        message_version: int,
        payload_hash: str,
    ) -> ContinuationDurabilityResult:
        """Require exact local intent durability and configured portable projection."""
        source_db = self.persistence.db if self.persistence is not None else None
        reader = getattr(source_db, "read_committed_chat_sync_intent", None)
        if not callable(reader):
            return ContinuationDurabilityResult(
                False,
                "Local continuation storage is unavailable; save the message and retry.",
            )
        source_record = reader(
            message_id=message_id,
            message_version=message_version,
            payload_hash=payload_hash,
        )
        if source_record is None:
            return ContinuationDurabilityResult(
                False,
                "Local continuation intent is stale or unavailable; save and retry.",
            )
        if self.sync_v2_server_profile_id is None:
            return ContinuationDurabilityResult(True, "local_intent_durable")

        producer = self.sync_v2_chat_producer
        if producer is None:
            return ContinuationDurabilityResult(
                False,
                "Portable sync projection is unavailable; restore sync configuration.",
            )
        repository = getattr(producer, "state_repository", None)
        if repository is None or getattr(repository, "is_durable", False) is not True:
            return ContinuationDurabilityResult(
                False,
                "Portable sync needs a file-backed state repository; configure one and retry.",
            )
        if getattr(producer, "source", None) is not source_db:
            return ContinuationDurabilityResult(
                False,
                "Portable sync source does not match local continuation storage.",
            )
        reconcile = getattr(producer, "reconcile_chat_message_intent", None)
        if not callable(reconcile):
            return ContinuationDurabilityResult(
                False,
                "Portable sync projection is unavailable; restore sync configuration.",
            )
        try:
            projection = reconcile(
                server_profile_id=self.sync_v2_server_profile_id,
                authenticated_principal_id=self.sync_v2_authenticated_principal_id,
                workspace_scope=self.sync_v2_workspace_scope,
                message_id=message_id,
                message_version=message_version,
                payload_hash=payload_hash,
            )
            profile = repository.get_sync_v2_profile_state(
                server_profile_id=self.sync_v2_server_profile_id,
                authenticated_principal_id=self.sync_v2_authenticated_principal_id,
                workspace_scope=self.sync_v2_workspace_scope,
            )
            dataset_id = profile.get("dataset_id") if profile else None
            persisted_receipt = (
                repository.get_sync_v2_source_projection_receipt(
                    server_profile_id=self.sync_v2_server_profile_id,
                    authenticated_principal_id=self.sync_v2_authenticated_principal_id,
                    workspace_scope=self.sync_v2_workspace_scope,
                    dataset_id=dataset_id,
                    domain="chat",
                    source_entity_id=message_id,
                    source_version=message_version,
                    source_payload_hash=payload_hash,
                )
                if dataset_id
                else None
            )
        except Exception:
            return ContinuationDurabilityResult(
                False,
                "Portable sync projection failed; restore local sync state and retry.",
            )
        projected_receipt = (
            projection.get("receipt") if isinstance(projection, Mapping) else None
        )
        if (
            not isinstance(projection, Mapping)
            or projection.get("status") != "enqueued"
            or not isinstance(projected_receipt, Mapping)
            or persisted_receipt is None
            or projected_receipt.get("client_envelope_id")
            != persisted_receipt.get("client_envelope_id")
            or persisted_receipt.get("source_entity_id") != message_id
            or persisted_receipt.get("source_version") != message_version
            or persisted_receipt.get("source_payload_hash") != payload_hash
        ):
            return ContinuationDurabilityResult(
                False,
                "Portable sync receipt is missing or inconsistent; reconcile and retry.",
            )
        return ContinuationDurabilityResult(True, "portable_projection_durable")

    def persist_provider_continuation_event(
        self,
        event: ProviderContinuationEvent,
    ) -> None:
        """Commit one runtime continuation event before its next side effect.

        The runtime callback is synchronous because it is invoked on the
        controller's existing agent worker thread. Persistent primary runs
        use the schema-v37 dedicated create/update operations; explicitly
        ephemeral runs retain no checkpoint and remain non-resumable.

        Args:
            event: Typed lifecycle event emitted by the shared agent runtime.

        Raises:
            RuntimeError: If ownership, optimistic state, persistence, or the
                durability barrier cannot be proven without private details.
        """
        context = event.context
        if context.durability == "ephemeral":
            return
        if context.agent_kind != "primary" or not context.owner_message_id:
            raise RuntimeError("Durable continuation needs a distinct assistant owner.")
        message = self._message_or_raise(context.owner_message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise RuntimeError("Durable continuation owner is unavailable.")
        persistence = self.persistence
        database = getattr(persistence, "db", None) if persistence is not None else None
        if database is None:
            raise RuntimeError(
                "Provider continuation could not be saved; retry or discard the interrupted run."
            )

        checkpoint, content = self._continuation_event_value(message, event)
        private_json = dump_provider_continuation_json(checkpoint)
        if private_json is None:
            raise RuntimeError("Durable continuation state is unavailable.")

        session_id = self._message_session_index[message.id]
        dispatch_recovery = self.dispatch_recovery_for_session(session_id)
        if (
            dispatch_recovery is not None
            and dispatch_recovery.assistant_message_id == message.id
            and dispatch_recovery.checkpoint is not None
        ):
            dispatch_checkpoint = dispatch_recovery.checkpoint
            if isinstance(event, ToolBatchReady):
                repository = getattr(persistence, "console_dispatch_repository", None)
                handoff = getattr(repository, "handoff_to_provider_continuation", None)
                if not callable(handoff):
                    self.mark_dispatch_recovery_needed(session_id, message.id)
                    raise RuntimeError("Provider continuation handoff is unavailable.")
                try:
                    result = handoff(
                        ConsoleContinuationHandoff(
                            assistant_message_id=message.persisted_message_id
                            or message.id,
                            expected_checkpoint_revision=(
                                dispatch_checkpoint.checkpoint_revision
                            ),
                            expected_user_message_version=(
                                dispatch_checkpoint.user_message_version
                            ),
                            expected_assistant_message_version=(
                                dispatch_checkpoint.assistant_message_version
                            ),
                            provider_continuation_json=private_json,
                        )
                    )
                except Exception as exc:
                    self.mark_dispatch_recovery_needed(session_id, message.id)
                    raise RuntimeError("Provider continuation handoff failed.") from exc
                if (
                    result.status is not ConsoleDispatchResultStatus.COMMITTED
                    or type(result.committed_message_version) is not int
                    or type(result.committed_payload_hash) is not str
                ):
                    self.mark_dispatch_recovery_needed(session_id, message.id)
                    raise RuntimeError(
                        "Provider continuation handoff conflicted; reload and retry."
                    )
                message.provider_continuation = checkpoint
                message.provider_continuation_message_version = (
                    result.committed_message_version
                )
                message.provider_continuation_remote = False
                message.provider_continuation_actions_enabled = True
                message.assistant_generation_state = "continuation_active"
                message.content = content
                with self._preparation_lock:
                    self._dispatch_recoveries_by_session.pop(session_id, None)
                    self._dispatch_recovery_message_baselines.pop(session_id, None)
                    self._dispatch_recovery_queue_hydration_pending.discard(session_id)
                durability = self.ensure_provider_continuation_durable(
                    message_id=message.persisted_message_id or message.id,
                    message_version=result.committed_message_version,
                    payload_hash=result.committed_payload_hash,
                )
                if not durability.ready:
                    message.provider_continuation_warning = durability.reason
                    raise RuntimeError(durability.reason)
                message.provider_continuation_warning = None
                return
            if isinstance(event, FinalContinuation):
                if not self.settle_dispatch_recovery(
                    session_id,
                    assistant_message_id=message.id,
                    terminal_state="complete",
                    content=content,
                    provider_continuation_json=private_json,
                    provider_continuation=checkpoint,
                ):
                    self.mark_dispatch_recovery_needed(session_id, message.id)
                    raise RuntimeError(
                        "Provider continuation terminal settlement failed."
                    )
                return

        message_version: int | None
        if message.persisted_message_id is None:
            if not isinstance(event, (ToolBatchReady, FinalContinuation)) or (
                event.expected_checkpoint_revision is not None
            ):
                raise RuntimeError("Durable continuation owner is unavailable.")
            session_id = self._message_session_index[message.id]
            conversation_id = self.persist_session_if_needed(session_id)
            if conversation_id is None:
                raise RuntimeError(
                    "Provider continuation could not be saved; retry or discard the interrupted run."
                )
            creator = getattr(database, "create_assistant_with_continuation", None)
            if not callable(creator):
                raise RuntimeError("Durable continuation storage is unavailable.")
            creator(
                message_id=message.id,
                conversation_id=conversation_id,
                parent_message_id=self._previous_persisted_message_id(message),
                content=content,
                provider_continuation_json=private_json,
            )
            message.persisted_message_id = message.id
            self._pending_persistence_message_ids.discard(message.id)
            message_version = 1
        else:
            version_reader = getattr(persistence, "get_message_version", None)
            durable_version = (
                version_reader(message.persisted_message_id)
                if callable(version_reader)
                else None
            )
            message_version = message.provider_continuation_message_version
            if message_version is None:
                message_version = durable_version
            if durable_version != message_version:
                raise RuntimeError("Continuation version conflict; reload and retry.")
            updater = getattr(database, "update_provider_continuation", None)
            if type(message_version) is not int or not callable(updater):
                raise RuntimeError("Durable continuation version is unavailable.")
            updater(
                message_id=message.persisted_message_id,
                expected_message_version=message_version,
                provider_continuation_json=private_json,
                content=content,
                assistant_generation_state=(
                    "continuation_active"
                    if checkpoint.state == "active"
                    else "complete"
                ),
            )
            message_version += 1

        message.provider_continuation = checkpoint
        message.provider_continuation_message_version = message_version
        message.provider_continuation_remote = False
        message.provider_continuation_actions_enabled = True
        message.assistant_generation_state = (
            "continuation_active" if checkpoint.state == "active" else "complete"
        )
        message.content = content
        payload = {
            "assistant_generation_state": message.assistant_generation_state,
            "content": content,
            "provider_continuation_json": private_json,
            "role": ConsoleMessageRole.ASSISTANT.value,
        }
        from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash

        durability = self.ensure_provider_continuation_durable(
            message_id=message.persisted_message_id,
            message_version=message_version,
            payload_hash=canonical_payload_hash(payload),
        )
        if not durability.ready:
            raise RuntimeError(durability.reason)
        message.provider_continuation_warning = None

    def provider_continuation_terminal_message(
        self,
        message_id: str,
        *,
        expected_content: str,
    ) -> ConsoleChatMessage | None:
        """Return an exactly persisted terminal event owner without rewriting it."""

        if type(expected_content) is not str:
            return None
        try:
            message = self._message_or_raise(message_id)
        except KeyError:
            return None
        checkpoint = message.provider_continuation
        version = message.provider_continuation_message_version
        session_id = self._message_session_index.get(message.id)
        if (
            message.role is not ConsoleMessageRole.ASSISTANT
            or message.persisted_message_id is None
            or message.status != "complete"
            or message.assistant_generation_state != "complete"
            or message.content != expected_content
            or checkpoint is None
            or checkpoint.state != "complete"
            or not checkpoint.rounds
            or checkpoint.rounds[-1].assistant_content != expected_content
            or type(version) is not int
            or version <= 0
            or message.provider_continuation_actions_enabled
            or session_id is None
            or self.dispatch_recovery_for_session(session_id) is not None
        ):
            return None
        database = getattr(self.persistence, "db", None) if self.persistence else None
        getter = getattr(database, "get_message_by_id", None)
        if not callable(getter):
            return None
        try:
            row = getter(message.persisted_message_id)
        except Exception:
            return None
        if not isinstance(row, Mapping):
            return None
        durable = read_provider_continuation_json(row.get("provider_continuation_json"))
        if (
            row.get("role") != ConsoleMessageRole.ASSISTANT.value
            or row.get("deleted") != 0
            or row.get("version") != version
            or row.get("assistant_generation_state") != "complete"
            or str(row.get("content") or "") != expected_content
            or durable.checkpoint != checkpoint
        ):
            return None
        return self._snapshot(message)

    @staticmethod
    def _continuation_event_value(
        message: ConsoleChatMessage,
        event: ProviderContinuationEvent,
    ) -> tuple[ProviderContinuationCheckpoint, str]:
        """Resolve a typed event without logging its private payload."""
        current = message.provider_continuation
        if isinstance(event, ToolBatchReady):
            if event.expected_checkpoint_revision is None:
                if current is not None:
                    raise RuntimeError("Continuation revision conflict.")
            elif (
                current is None
                or current.checkpoint_revision != event.expected_checkpoint_revision
            ):
                raise RuntimeError("Continuation revision conflict.")
            return event.checkpoint, message.content
        if isinstance(event, FinalContinuation):
            if event.expected_checkpoint_revision is None:
                if current is not None:
                    raise RuntimeError("Continuation revision conflict.")
            elif (
                current is None
                or current.checkpoint_revision != event.expected_checkpoint_revision
            ):
                raise RuntimeError("Continuation revision conflict.")
            return event.checkpoint, event.assistant_content
        if current is None:
            raise RuntimeError("Durable continuation owner is unavailable.")
        if isinstance(event, ToolCallExecuting):
            checkpoint = transition_provider_call(
                current,
                call_id=event.call_id,
                expected_revision=event.expected_checkpoint_revision,
                target="executing",
            )
            return checkpoint, message.content
        if isinstance(event, ToolCallFinished):
            checkpoint = transition_provider_call(
                current,
                call_id=event.call_id,
                expected_revision=event.expected_checkpoint_revision,
                target=event.target_state,
                result=event.result,
            )
            return checkpoint, message.content
        raise RuntimeError("Unsupported continuation event.")

    def _enqueue_sync_v2_message_if_ready(
        self,
        message: ConsoleChatMessage,
        *,
        content_safe_diagnostic: bool = False,
    ) -> None:
        if (
            self.sync_v2_chat_producer is None
            or self.sync_v2_server_profile_id is None
            or message.persisted_message_id is None
            or message.status != "complete"
            or not message.content
        ):
            return
        session_id = self._message_session_index.get(message.id)
        if session_id is None:
            return
        session = self._sessions.get(session_id)
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
        if conversation_id is None:
            return
        variant_metadata = self._sync_variant_metadata(message)
        stable_key = f"{conversation_id}:{message.persisted_message_id}"
        try:
            result = self.sync_v2_chat_producer.enqueue_chat_message(
                server_profile_id=self.sync_v2_server_profile_id,
                authenticated_principal_id=self.sync_v2_authenticated_principal_id,
                workspace_scope=self.sync_v2_workspace_scope,
                conversation_id=conversation_id,
                message_id=message.persisted_message_id,
                role=message.role.value,
                content=message.content,
                parent_message_id=self._previous_persisted_message_id(message),
                sequence=self._sync_message_sequence(message),
                variant_turn_id=variant_metadata["variant_turn_id"],
                variant_index=variant_metadata["variant_index"],
                variant_count=variant_metadata["variant_count"],
                selected_variant_id=variant_metadata["selected_variant_id"],
                base_version=self._sync_v2_message_versions.get(stable_key),
                entity_version=None,
            )
            self._record_sync_v2_message_version(stable_key, result)
        except Exception:
            if content_safe_diagnostic:
                logger.warning("terminal_persistence_bookkeeping_unavailable")
                return
            logger.bind(
                server_profile_id=self.sync_v2_server_profile_id,
                authenticated_principal_id=self.sync_v2_authenticated_principal_id,
                workspace_scope=self.sync_v2_workspace_scope,
                conversation_id=conversation_id,
                message_id=message.persisted_message_id,
            ).exception("Failed to enqueue Sync v2 chat message after local mutation")

    def _project_sync_v2_message_deletes(
        self, tombstones: Iterable[Mapping[str, Any]]
    ) -> None:
        """Best-effort projection of already-committed local tombstones."""
        producer = self.sync_v2_chat_producer
        profile_id = self.sync_v2_server_profile_id
        reconcile = getattr(producer, "reconcile_chat_message_delete_intent", None)
        if profile_id is None or not callable(reconcile):
            return
        payload_hash = canonical_payload_hash({"deleted": True})
        for tombstone in tombstones:
            try:
                reconcile(
                    server_profile_id=profile_id,
                    authenticated_principal_id=self.sync_v2_authenticated_principal_id,
                    workspace_scope=self.sync_v2_workspace_scope,
                    message_id=str(tombstone["message_id"]),
                    message_version=int(tombstone["version"]),
                    payload_hash=payload_hash,
                )
            except Exception:
                logger.warning(
                    "Failed to project Sync v2 Chat tombstone after local mutation"
                )

    def _sync_message_sequence(self, message: ConsoleChatMessage) -> int | None:
        """Return ``message``'s 1-based sync-eligible position on the active path.

        Tree-aware (Task 5): ``_messages_by_session[session_id]`` is no
        longer a flat append-order history of every message ever created --
        since Task 3 it is the derived active-path VIEW (root -> active
        leaf), rebuilt by ``_recompute_active_path`` alone. Counting along it
        therefore already counts along the current branch rather than across
        every fork, which is what a sequence number for the visible
        conversation should mean. A message that is currently off the active
        path (e.g. an old sibling left behind by ``create_sibling``, or any
        node reached only via ``get_message``/``select_variant`` while
        another branch is active) is not found in this walk and returns
        ``None``, same as before.
        """
        session_id = self._message_session_index.get(message.id)
        if session_id is None:
            return None
        sequence = 0
        for candidate in self._messages_by_session.get(session_id, []):
            if self._is_sync_eligible_message(candidate):
                sequence += 1
            if candidate.id == message.id:
                return sequence if self._is_sync_eligible_message(candidate) else None
        return None

    @staticmethod
    def _is_sync_eligible_message(message: ConsoleChatMessage) -> bool:
        return (
            message.persisted_message_id is not None
            and message.status == "complete"
            and bool(message.content)
        )

    def _previous_persisted_message_id(self, message: ConsoleChatMessage) -> str | None:
        """Return the persisted id of ``message``'s nearest PERSISTED ancestor.

        Tree-aware (Task 5): previously this walked the flat message list
        looking for whatever came immediately before ``message`` with a
        persisted id -- a linear-history assumption that breaks the moment a
        branch forks (a sibling's "previous" message is not "whatever this
        session last appended", it's specifically the shared parent).

        Resolving the nearest persisted ancestor via
        ``_nearest_persisted_ancestor_id`` (skipping non-persisted mid-chain
        nodes) fixes the fork case AND keeps the Sync v2 parent connected
        across a ``persist=False`` interstitial. Note this exactly restores
        the OLD flat-list behavior for the interstitial case -- that walk also
        skipped non-persisted messages -- and for a plain linear conversation
        the immediate parent IS the nearest persisted ancestor, so the value
        is unchanged. ``None`` when no ancestor is persisted (root, unknown
        session, or nothing durably persisted above yet).
        """
        session_id = self._message_session_index.get(message.id)
        if session_id is None:
            return None
        return self._nearest_persisted_ancestor_id(session_id, message)

    @staticmethod
    def _sync_variant_metadata(
        message: ConsoleChatMessage,
    ) -> dict[str, str | int | None]:
        if message.variants is None:
            return {
                "variant_turn_id": None,
                "variant_index": None,
                "variant_count": None,
                "selected_variant_id": None,
            }
        return {
            "variant_turn_id": message.variants.turn_id,
            "variant_index": message.variants.selected_index,
            "variant_count": len(message.variants.variants),
            "selected_variant_id": message.variants.current.id,
        }

    def _record_sync_v2_message_version(
        self, stable_key: str, result: dict[str, Any]
    ) -> None:
        if result.get("status") != "enqueued":
            return
        entry = result.get("outbox_entry")
        if not isinstance(entry, dict):
            return
        envelope = entry.get("envelope")
        if not isinstance(envelope, dict):
            return
        payload_hash = envelope.get("payload_hash")
        if isinstance(payload_hash, str) and payload_hash:
            self._sync_v2_message_versions[stable_key] = payload_hash

    def _session_or_raise(self, session_id: str) -> ConsoleChatSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown Console chat session: {session_id}") from exc

    def _message_or_raise(self, message_id: str) -> ConsoleChatMessage:
        # Resolve from the FULL tree, not the active-path view, so off-path
        # nodes (siblings of the active branch) are findable. Display-only TOOL
        # markers are intentionally NOT tree nodes, so they do not resolve here.
        session_id = self._message_session_index.get(message_id)
        if session_id is None:
            raise KeyError(f"Unknown Console message: {message_id}")
        node = self._nodes_by_session.get(session_id, {}).get(message_id)
        if node is not None:
            return node
        raise KeyError(f"Unknown Console message: {message_id}")

    @staticmethod
    def _selected_speech_variant_id(message: ConsoleChatMessage) -> str:
        """Return the native linear id or exact selected text-variant id."""
        if message.variants is None:
            return message.id
        try:
            selected_id = message.variants.current.id
        except (AttributeError, IndexError):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            ) from None
        if type(selected_id) is not str or not selected_id:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        return selected_id

    @classmethod
    def _speech_selection(
        cls,
        message: ConsoleChatMessage,
    ) -> tuple[str, str]:
        """Return exact visible text and its selected text-variant identity."""
        selected_variant_id = cls._selected_speech_variant_id(message)
        if type(message.content) is not str:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
            )
        if message.variants is not None:
            try:
                selected_content = message.variants.current.content
            except (AttributeError, IndexError):
                raise ConsoleSpeechSnapshotRejected(
                    ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
                ) from None
            if type(selected_content) is not str or message.content != selected_content:
                raise ConsoleSpeechSnapshotRejected(
                    ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
                )
        return message.content, selected_variant_id

    def _persisted_message_version_or_reject(
        self,
        message: ConsoleChatMessage,
    ) -> int | None:
        """Read the durable version fence or reject an unverifiable row."""
        persisted_message_id = message.persisted_message_id
        if persisted_message_id is None:
            return None
        if type(persisted_message_id) is not str or not persisted_message_id:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.PERSISTED_VERSION_UNAVAILABLE
            )
        persistence = self.persistence
        reader = (
            getattr(persistence, "get_message_version", None)
            if persistence is not None
            else None
        )
        if not callable(reader):
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.PERSISTED_VERSION_UNAVAILABLE
            )
        try:
            version = reader(persisted_message_id)
        except Exception:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.PERSISTED_VERSION_UNAVAILABLE
            ) from None
        if type(version) is not int or version < 1:
            raise ConsoleSpeechSnapshotRejected(
                ConsoleSpeechSnapshotRejectionCode.PERSISTED_VERSION_UNAVAILABLE
            )
        return version

    def _bump_message_speech_revision(self, message_id: str) -> None:
        """Advance one registered node's process-local speech fence."""
        self._message_speech_revisions[message_id] += 1

    def _bump_payload_revision(self, session_id: str) -> None:
        """Mark the session's provider payload as changed (cost-ticker PR3).

        Bumped at every mutation that can change what a future send would
        transmit; the cost chip recomputes its cache-break fingerprint only
        when this moves, so a missed bump means a stale chip (annoying), not
        a wrong send (impossible from here).
        """
        self._payload_revisions[session_id] = (
            self._payload_revisions.get(session_id, 0) + 1
        )

    def get_payload_revision(self, session_or_conversation_id: str) -> int:
        """Return the payload-revision counter for a session or conversation.

        Public read side of the revision bus the trajectory live view polls
        (task-5): the counter has no observer interface, so the screen
        ``set_interval``-polls this getter. Accepts either a Console session
        id or a persisted conversation id -- the trajectory write path bumps
        BOTH keys -- returning the newest of the matches (0 when unknown).
        """
        revision = self._payload_revisions.get(session_or_conversation_id, 0)
        for session in self._sessions.values():
            if session.persisted_conversation_id == session_or_conversation_id:
                revision = max(revision, self._payload_revisions.get(session.id, 0))
        return revision

    def _bump_conversation_context_epoch(self, session_id: str) -> None:
        """Advance one live session's provider-context change token."""
        self._conversation_context_epochs[session_id] += 1

    def conversation_context_epoch(self, session_id: str) -> int:
        """Return the process-local provider-context epoch for a live session.

        Unlike ``payload_revision``, ordinary linear appends, response streaming,
        terminal status, and persistence bookkeeping do not advance this token.
        It exists for deferred-turn safety and is never serialized.

        Raises:
            KeyError: If ``session_id`` is not a live Console session.
        """
        self._session_or_raise(session_id)
        return self._conversation_context_epochs[session_id]

    def _message_is_on_active_path(self, message_id: str) -> bool:
        """Return whether a registered tree node affects the active transcript."""
        session_id = self._message_session_index[message_id]
        return message_id in self.active_path_message_ids(session_id)

    def _settle_failed_retry_context(
        self,
        message: ConsoleChatMessage,
        *,
        provider_visible: bool,
    ) -> None:
        """Finish failed-row retry tracking and advance on visible recovery.

        A failed row is excluded from normal future payloads. Once an in-place
        retry finishes as complete or stopped, that same row becomes history and
        therefore changes effective provider context even when its text is byte-
        identical to the failed attempt. Another failed terminal stays excluded.
        """
        was_failed_retry = message.id in self._failed_retry_message_ids
        self._failed_retry_message_ids.discard(message.id)
        if not was_failed_retry or not provider_visible:
            return
        if self._message_is_on_active_path(message.id):
            session_id = self._message_session_index[message.id]
            self._bump_conversation_context_epoch(session_id)

    def payload_revision(self, session_id: str) -> int:
        """Monotonic per-session counter of payload-affecting mutations."""
        return self._payload_revisions.get(session_id, 0)

    def _register_tree_node(
        self,
        session_id: str,
        message: ConsoleChatMessage,
        *,
        parent_native_id: str | None,
    ) -> None:
        """Register a real message as a node in ALL tree structures.

        The ONE place a node enters ``_nodes_by_session``,
        ``_children_by_parent``, ``_native_parent_by_message``, and
        ``_message_session_index`` together, so every registration path stays
        consistent. Does NOT set the active leaf or recompute the view -- the
        caller owns leaf placement and the follow-up ``_recompute_active_path``.
        """
        self._nodes_by_session.setdefault(session_id, {})[message.id] = message
        self._native_parent_by_message[message.id] = parent_native_id
        self._children_by_parent.setdefault(session_id, {}).setdefault(
            parent_native_id, []
        ).append(message.id)
        self._message_session_index[message.id] = session_id
        self._message_speech_revisions[message.id] = 0
        self._message_completion_generations[message.id] = 0

    def _ingest_linear_messages(
        self, session_id: str, messages: Iterable[ConsoleChatMessage]
    ) -> None:
        """Register a flat message list as a linear tree chain, then recompute.

        Used by the restore paths (``restore_state`` /
        ``restore_persisted_session``): each real message is parented at the
        previous real message, the last real message becomes the active leaf,
        and ``_recompute_active_path`` reproduces the exact restored list. TOOL
        markers, being display-only, are never registered as tree nodes (they
        would be dropped by the immediate recompute anyway -- the accepted
        Phase A limitation; restore inputs do not carry them in practice).
        """
        parent_native_id: str | None = None
        for message in messages:
            restored = replace(
                message,
                citation_presentation=None,
                activity_presentation=None,
            )
            if restored.role is ConsoleMessageRole.TOOL:
                self._message_session_index[restored.id] = session_id
                continue
            self._register_tree_node(
                session_id, restored, parent_native_id=parent_native_id
            )
            parent_native_id = restored.id
        self._active_leaf_by_session[session_id] = parent_native_id
        self._recompute_active_path(session_id)

    def _ingest_full_tree(
        self,
        session_id: str,
        all_nodes: Iterable[ConsoleChatMessage],
        *,
        active_leaf_persisted_id: str | None,
    ) -> None:
        """Rebuild the FULL conversation tree (all branches) from persisted nodes.

        Task 8 resume path. ``all_nodes`` is the flattened persisted tree in
        pre-order (every node, siblings in the DB's timestamp order); each node
        carries its own ``persisted_message_id`` and its persisted
        ``parent_message_id``. The tree is reconnected by mapping persisted ids
        to fresh native ids, so off-path siblings load as navigable nodes -- the
        whole point of Task 8. The active-path VIEW is then derived from the
        stored active-leaf pointer, falling back to the most-recent-child leaf
        (``children[-1]`` walk) and repairing the durable pointer when the
        pointer is ``None``, unknown, or dangling.

        TOOL markers (display-only, never tree nodes) are not expected in
        ``all_nodes`` -- resume re-derives them from ``AgentRunsDB`` and overlays
        them onto the view afterward -- but any that slip in are registered in
        the session index only, mirroring ``_ingest_linear_messages``.
        """
        registered: list[ConsoleChatMessage] = []
        persisted_to_native: dict[str, str] = {}
        for node in all_nodes:
            restored = replace(
                node,
                citation_presentation=None,
                activity_presentation=None,
            )
            if restored.role is ConsoleMessageRole.TOOL:
                self._message_session_index[restored.id] = session_id
                continue
            registered.append(restored)
            if restored.persisted_message_id is not None:
                # Last write wins on a (malformed) duplicate persisted id; the
                # tree is still internally consistent, just under-linked.
                persisted_to_native[restored.persisted_message_id] = restored.id
        for restored in registered:
            native_parent = persisted_to_native.get(restored.parent_message_id)
            self._register_tree_node(
                session_id, restored, parent_native_id=native_parent
            )
        # Legacy flat-data repair (C1): before branching, every message was
        # persisted with parent_message_id=NULL, so an existing conversation
        # loads as N separate roots (all siblings under None) with no children.
        # Chain them into one linear spine so the active-leaf walk traverses the
        # whole conversation instead of truncating to the last root.
        self._chain_legacy_flat_roots(session_id)
        # Resolve the active leaf from the stored pointer; fall back to the
        # most-recent leaf when it is missing/unknown/dangling, and repair the
        # durable pointer so the next resume is exact.
        leaf_native: str | None = None
        if active_leaf_persisted_id is not None:
            leaf_native = persisted_to_native.get(active_leaf_persisted_id)
        used_fallback = leaf_native is None
        if used_fallback:
            leaf_native = self._most_recent_leaf_native(session_id)
        self._active_leaf_by_session[session_id] = leaf_native
        self._recompute_active_path(session_id)
        if used_fallback and leaf_native is not None:
            # Map the fallback leaf back to its persisted id and write it
            # through (``_persist_active_leaf`` no-ops without a durable seam).
            self._persist_active_leaf(session_id, leaf_native)
        # Console `/rewind` (SP2): map the persisted context-summary boundary
        # back to a native id on the newly-loaded tree, fail-open (leave
        # unset) when the summary is absent or its boundary is dangling.
        self._resolve_context_summary_on_resume(session_id, persisted_to_native)

    def _chain_legacy_flat_roots(self, session_id: str) -> None:
        """Chain multiple root-level threads into one linear spine (C1 repair).

        Pre-feature Console persistence wrote EVERY message with
        ``parent_message_id=NULL`` (the base ``_persist_new_message`` hardcoded
        ``None``), so an existing conversation ``[U1, A1, U2, A2]`` is stored as
        four separate roots -- all siblings under ``None``, none with children.
        On resume the active-leaf fallback (``_most_recent_leaf_native``) then
        walks only the LAST root, collapsing the transcript to its final message
        and rendering a phantom ``n/n`` sibling counter on the survivor.

        Historically a GENUINE Console branch was ALWAYS a set of siblings
        under a shared *non-None* parent (regenerate / create-sibling parent
        the new node at the anchor's parent), NEVER two separate root
        threads -- a conversation's real root is its single first message.
        So more than one root-level thread meant legacy flat data (fully
        flat, or a flat prefix followed by post-feature branched messages),
        and it was always correct to chain the roots into a single linear
        spine.

        Phase B's ``edit_and_resend_message`` broke that invariant on
        purpose: editing-and-resending the conversation's very FIRST user
        message forks a NEW root-level USER sibling (``create_sibling``
        parents the fork at the anchor's own parent, which is ``None`` for a
        root message) -- a genuine branch that legitimately has more than one
        root thread. A genuine root-level fork's siblings are ALWAYS all USER
        (an ASSISTANT node's native parent is never ``None`` -- it always
        replies to a user turn, even the very first one), so a role-MIXED root
        set (both USER and ASSISTANT at the root) can ONLY be legacy flat data
        and is chained. Role-homogeneity is thus the distinguishing signal: a
        single-role (all-USER) root set is treated as a genuine Phase-B branch
        and left alone (chaining it would silently splice the newer branch onto
        the older as a fake parent-child link, corrupting the tree so a
        swipe/resume shows the wrong content).

        task-572 strengthens the fingerprint for the all-USER root set: a
        DEGENERATE legacy conversation whose 2+ user turns each got NO
        assistant reply (reachable in the flat era via repeated
        failed/blocked sends) also loads as all-USER roots -- but its roots
        are ALL CHILDLESS, whereas a genuine first-message edit-&-resend
        fork always hangs at least one reply subtree under a root (the
        anchor's old tail, and/or the resent branch's own reply). So an
        all-USER root set is chained when every root is childless, and left
        alone when any root has a subtree.

        RESIDUAL EDGE (not airtight, narrower than the pre-task-572 gap): a
        genuine first-message fork whose BOTH branches ended up childless --
        the anchor never got a reply AND the resent branch's reply never
        persisted (killed mid-stream before the first flushed chunk) -- is
        indistinguishable from degenerate legacy and now chains. Non-data-
        loss (both user rows stay visible, linearly), and strictly rarer
        than the all-USER-legacy shape this fixes; the two shapes are
        provably indistinguishable from the persisted tree alone, so no
        local heuristic can be perfect.

        Roots are chained in their existing insertion order, which is the DB's
        timestamp-ASC order (``get_root_messages_for_conversation`` orders roots
        by timestamp; ``ConsoleChatMessage`` carries no timestamp of its own, so
        insertion order is the ordering signal -- exactly the accepted fallback
        for equal/absent timestamps). Each root ``r_i`` (i >= 1) is re-parented
        onto ``r_{i-1}`` and moved out of the ``None`` bucket into
        ``r_{i-1}``'s ordered child list; any real subtree already hanging off a
        root (e.g. a post-feature message whose real parent is a flat row) is
        left intact. After chaining there is exactly one root (``r_0``) and the
        active-leaf ancestry walk traverses the full spine plus any subtrees.

        A single-root (genuine) tree is left untouched -- the chaining branch
        never triggers. This is an IN-MEMORY reconstruction only; durable
        ``parent_message_id`` rows are never rewritten (the active-leaf pointer
        repair on resume is the durable fix).
        """
        children = self._children_by_parent.get(session_id)
        if children is None:
            return
        roots = children.get(None, [])
        if len(roots) <= 1:
            return
        nodes = self._nodes_by_session.get(session_id, {})
        root_has_assistant = any(
            nodes[root_id].role is ConsoleMessageRole.ASSISTANT
            for root_id in roots
            if root_id in nodes
        )
        all_roots_childless = all(not children.get(root_id) for root_id in roots)
        if not root_has_assistant and not all_roots_childless:
            # All-USER roots with at least one reply subtree: a genuine
            # Phase-B root-level fork (an ASSISTANT node's parent is never
            # None, so any root assistant row is the legacy signature; and a
            # real fork always carries a subtree). Leave each root
            # independently navigable via `siblings_at`/`set_active_leaf`.
            return
        # Keep only the first root under None; chain the rest onto their
        # predecessor, preserving each root's own existing subtree.
        children[None] = [roots[0]]
        previous = roots[0]
        for root in roots[1:]:
            self._native_parent_by_message[root] = previous
            children.setdefault(previous, []).append(root)
            previous = root

    def _most_recent_leaf_native(self, session_id: str) -> str | None:
        """Return the deepest ``children[-1]`` leaf under the most-recent root.

        The fallback leaf resolver when a session has no usable active-leaf
        pointer. Roots (and children) are ordered oldest-first, so the last
        root and each step's last child track the most recently created branch
        -- the same branch the pre-pointer ``children[-1]`` resume walk showed.
        Returns ``None`` when the session has no tree nodes.
        """
        roots = self._children_by_parent.get(session_id, {}).get(None, [])
        if not roots:
            return None
        return self._leaf_under(roots[-1])

    def _recompute_active_path(self, session_id: str) -> None:
        """Rebuild the active-path VIEW for a session from live tree nodes.

        The SINGLE writer of ``_messages_by_session[session_id]``. Walks the
        active leaf up to the root via ``_native_parent_by_message``, reverses
        to root->leaf order, and materializes the view from the LIVE node
        objects in ``_nodes_by_session`` (never copies -- streaming mutates node
        content in place and the view must observe it). Each visited node's
        transient ``sibling_index``/``sibling_count`` is filled from its native
        parent's ordered child list so the renderer can show ``<``/``>`` + an
        ``n/m`` counter without reaching into store internals.

        A visited-set guards against a malformed cyclic parent chain (real
        DBs can't produce one -- unique PKs -- so this is defensive-only,
        mirroring ``_nearest_persisted_ancestor_id``).
        """
        nodes = self._nodes_by_session.get(session_id, {})
        children = self._children_by_parent.get(session_id, {})
        path_ids: list[str] = []
        visited: set[str] = set()
        current = self._active_leaf_by_session.get(session_id)
        while current is not None and current not in visited:
            visited.add(current)
            path_ids.append(current)
            current = self._native_parent_by_message.get(current)
        path_ids.reverse()
        path: list[ConsoleChatMessage] = []
        for native_id in path_ids:
            node = nodes.get(native_id)
            if node is None:
                continue
            siblings = children.get(self._native_parent_by_message.get(native_id), [])
            node.sibling_count = len(siblings)
            node.sibling_index = (
                siblings.index(native_id) if native_id in siblings else 0
            )
            path.append(node)
        self._messages_by_session[session_id] = self._with_tool_markers(
            session_id, path
        )

    def _with_tool_markers(
        self, session_id: str, path: list[ConsoleChatMessage]
    ) -> list[ConsoleChatMessage]:
        """Splice this session's TOOL markers back into a rebuilt path view.

        TASK-1842. Markers are display-only and never tree nodes, so the
        node walk in `_recompute_active_path` cannot see them. Each marker
        remembers the node it followed; it is re-inserted directly after that
        node so the trace reads in the order the agent produced it.

        A marker whose anchor is NOT on the current path is dropped, which is
        the correct behavior rather than a gap: the anchor is off-path
        because the user switched branches (regenerate / edit-and-resend), and
        those tool calls belong to the branch they were made on.

        Args:
            session_id: Session whose view is being rebuilt.
            path: The freshly walked node path, root -> leaf.

        Returns:
            The path with markers interleaved.
        """
        markers = self._tool_markers_by_session.get(session_id)
        if not markers:
            return path
        by_anchor: dict[str | None, list[ConsoleChatMessage]] = {}
        for anchor, marker in markers:
            by_anchor.setdefault(anchor, []).append(marker)
        merged: list[ConsoleChatMessage] = []
        # Markers recorded before any node existed (anchor None) lead the view.
        merged.extend(by_anchor.get(None, ()))
        for node in path:
            merged.append(node)
            merged.extend(by_anchor.get(node.id, ()))
        return merged

    def _purge_tool_markers(self, session_id: str, anchors: set[str]) -> None:
        """Drop this session's markers anchored to any node id in ``anchors``.

        TASK-1842 follow-up. Markers are display-only, so the node sweeps in
        `delete_message` cannot reach them: deleting the branch a marker
        followed left the marker object registered and its id still in
        `_message_session_index`, claiming the session owned a message it
        could never render again (`_with_tool_markers` drops off-path anchors).

        Args:
            session_id: Session whose marker registry is being pruned.
            anchors: Native node ids being removed; markers anchored to any of
                them go with the node they belonged to.
        """
        markers = self._tool_markers_by_session.get(session_id)
        if not markers:
            return
        kept: list[tuple[str | None, ConsoleChatMessage]] = []
        for anchor, marker in markers:
            if anchor is not None and anchor in anchors:
                self.clear_terminal_citation_state(marker.id)
                self._message_session_index.pop(marker.id, None)
                continue
            kept.append((anchor, marker))
        if kept:
            self._tool_markers_by_session[session_id] = kept
        else:
            self._tool_markers_by_session.pop(session_id, None)

    def _subtree_ids(self, session_id: str, root_id: str) -> list[str]:
        """Return ``root_id`` plus all its descendant native ids (pre-order)."""
        children_map = self._children_by_parent.get(session_id, {})
        collected: list[str] = []
        stack = [root_id]
        while stack:
            node_id = stack.pop()
            collected.append(node_id)
            stack.extend(children_map.get(node_id, []))
        return collected

    def _tree_nodes_parent_first(self, session_id: str) -> list[ConsoleChatMessage]:
        """Return EVERY tree node for a session, guaranteed parent-before-child.

        Used by ``promote_ephemeral_session`` (task-3), which must persist the
        whole conversation tree -- every off-path branch left behind by
        ``create_sibling`` (regenerate / edit-and-resend), not just the
        active-path view -- so a promoted temporary chat comes out
        indistinguishable from one that had been saved from the start,
        swipe-back included.

        Ordering is load-bearing, not cosmetic: ``_persist_new_message``
        resolves each node's persisted parent via
        ``_nearest_persisted_ancestor_id``, which walks up
        ``_native_parent_by_message`` looking for the nearest ANCESTOR that
        already has a ``persisted_message_id``. Persisting a child before its
        parent would leave that walk with nothing to find (a stray root) or,
        worse, silently resolve to some unrelated already-persisted ancestor
        further up the chain -- a misparented row that looks fine until a
        later resume walks the wrong branch.

        A breadth-first walk from the roots (``_children_by_parent[session_id]
        [None]``) down through ``_children_by_parent`` guarantees this by
        construction: a node is only enqueued once its parent has already been
        dequeued and emitted. This does NOT rely on ``_nodes_by_session``'s
        dict insertion/iteration order for correctness -- that order is
        unspecified by this method's contract even though CPython dicts
        happen to preserve insertion order today.

        Returns:
            Every node, in an order where each node's parent (if any)
            precedes it. TOOL markers are excluded -- they are display-only
            and never become tree nodes (see ``_register_tree_node``).
        """
        nodes = self._nodes_by_session.get(session_id, {})
        children_map = self._children_by_parent.get(session_id, {})
        ordered: list[ConsoleChatMessage] = []
        queue: deque[str] = deque(children_map.get(None, []))
        while queue:
            node_id = queue.popleft()
            node = nodes.get(node_id)
            if node is not None:
                ordered.append(node)
            queue.extend(children_map.get(node_id, []))
        return ordered

    def _leaf_under(self, node_id: str) -> str:
        """Return the deepest descendant of ``node_id`` (always the last child).

        Used by later swipe/select tasks to resolve which leaf a sibling switch
        should land on. Walks ``_children_by_parent`` picking the last child at
        each step; returns ``node_id`` itself when it has no children.
        """
        session_id = self._message_session_index.get(node_id)
        children_map = (
            self._children_by_parent.get(session_id, {}) if session_id else {}
        )
        current = node_id
        while True:
            children = children_map.get(current)
            if not children:
                return current
            current = children[-1]

    def _persist_active_leaf(
        self,
        session_id: str,
        message_id: str | None,
        *,
        content_safe_diagnostic: bool = False,
    ) -> None:
        """Write-through the local-only active-leaf pointer for a persisted conv.

        No-op unless the session owns a persisted conversation AND the
        persistence adapter exposes a raw ``db`` seam (mirrors the
        ``persistence_db = getattr(self.persistence, "db", None)`` pattern in
        ``persist_session_if_needed``). Maps the in-memory leaf to its persisted
        message id (``None`` when cleared or not yet persisted).
        """
        session = self._sessions.get(session_id)
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
        if conversation_id is None:
            return
        persistence_db = getattr(self.persistence, "db", None)
        if persistence_db is None:
            return
        leaf_persisted_id: str | None = None
        if message_id is not None:
            node = self._nodes_by_session.get(session_id, {}).get(message_id)
            leaf_persisted_id = node.persisted_message_id if node is not None else None
        try:
            persistence_db.set_conversation_active_leaf(
                conversation_id, leaf_persisted_id
            )
        except Exception:
            if content_safe_diagnostic:
                logger.warning("terminal_persistence_bookkeeping_unavailable")
                return
            logger.bind(
                session_id=session_id,
                conversation_id=conversation_id,
            ).exception(
                "Failed to persist Console active-leaf pointer; the in-memory "
                "pointer keeps the applied value."
            )

    def _persist_context_summary(
        self,
        session_id: str,
        summary: str | None,
        boundary_native_id: str | None,
    ) -> None:
        """Write-through the local-only context-summary pair for a persisted conv.

        Console `/rewind` "summarize up to here" (SP2). No-op unless the
        session owns a persisted conversation AND the persistence adapter
        exposes a raw ``db`` seam (mirrors ``_persist_active_leaf``). Maps
        the in-memory boundary to its persisted message id (``None`` when
        cleared or not yet persisted).
        """
        session = self._sessions.get(session_id)
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
        if conversation_id is None:
            return
        persistence_db = getattr(self.persistence, "db", None)
        if persistence_db is None:
            return
        boundary_persisted_id: str | None = None
        if boundary_native_id is not None:
            node = self._nodes_by_session.get(session_id, {}).get(boundary_native_id)
            boundary_persisted_id = (
                node.persisted_message_id if node is not None else None
            )
        try:
            persistence_db.set_conversation_context_summary(
                conversation_id, summary, boundary_persisted_id
            )
        except Exception:
            logger.bind(
                session_id=session_id,
                conversation_id=conversation_id,
            ).exception(
                "Failed to persist Console context-summary; the in-memory "
                "pair keeps the applied value."
            )

    def _resolve_context_summary_on_resume(
        self, session_id: str, persisted_to_native: dict[str, str]
    ) -> None:
        """Map the persisted context-summary boundary back to a native id.

        Console `/rewind` resume mapping (SP2). Reads
        ``get_conversation_context_summary`` via the db seam (mirrors
        ``_persist_active_leaf``'s ``getattr(self.persistence, "db", None)``
        guard) -- no-op unless the session owns a persisted conversation and
        the persistence adapter exposes the seam. The stored boundary is a
        *persisted* message id; when it maps to a node on the just-loaded
        tree (``persisted_to_native``, built by ``_ingest_full_tree``), the
        in-memory summary state is set. Absent, unreadable, or dangling (no
        stored summary, or a boundary id not present on the loaded tree)
        leaves the in-memory state unset -- fail-open, matching the design's
        rule that an inert/dangling boundary falls back to full history. A
        DANGLING boundary additionally best-effort clears the stale
        persisted pair (``set_conversation_context_summary(conversation_id,
        None, None)``, guarded + exception-swallowed like the write-through
        above) so a permanently-orphaned boundary (e.g. its branch was
        hard-deleted, or a foreign client rewrote history) doesn't linger in
        the DB row indefinitely -- benign either way (this path already
        fails open, and the next summarize overwrites it), just misleading
        to anything that reads the column directly.
        """
        session = self._sessions.get(session_id)
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
        if conversation_id is None:
            return
        persistence_db = getattr(self.persistence, "db", None)
        if persistence_db is None:
            return
        try:
            summary, boundary_persisted_id = (
                persistence_db.get_conversation_context_summary(conversation_id)
            )
        except Exception:
            logger.bind(
                session_id=session_id,
                conversation_id=conversation_id,
            ).exception(
                "Failed to read Console context-summary on resume; leaving unset."
            )
            return
        if summary is None or boundary_persisted_id is None:
            return
        boundary_native_id = persisted_to_native.get(boundary_persisted_id)
        if boundary_native_id is None:
            # Dangling: the persisted boundary message isn't on the loaded
            # tree. Leave the in-memory state unset (fail-open) AND
            # best-effort clear the now-permanently-orphaned persisted pair
            # so it doesn't linger indefinitely -- non-fatal, mirrors
            # ``_persist_context_summary``'s write-through guard.
            try:
                persistence_db.set_conversation_context_summary(
                    conversation_id, None, None
                )
            except Exception:
                logger.bind(
                    session_id=session_id,
                    conversation_id=conversation_id,
                ).exception(
                    "Failed to clear stale Console context-summary with a "
                    "dangling boundary; the persisted pair may linger."
                )
            return
        self._context_summary_by_session[session_id] = (summary, boundary_native_id)

    def _fold_stream_buffer_without_persistence(
        self,
        message: ConsoleChatMessage,
    ) -> bool:
        """Fold buffered stream chunks into ``message.content`` without a write.

        TASK-259: after joining, the chunk list is collapsed to the single
        joined string (in place, preserving any outstanding list references),
        so the next materialize joins only the chunks that arrived since this
        one instead of re-walking the whole stream history on every 0.2s
        tick. The invariant ``"".join(buffer) == full streamed content`` is
        preserved for every reader.

        Args:
            message: Store-owned message whose visible content should
                reflect all chunks appended so far.

        Returns:
            Whether new chunks were folded into ``message.content``.
        """
        buffer = self._stream_chunks_by_message.get(message.id)
        if not buffer:
            return False
        if self._stream_materialized_counts.get(message.id) == len(buffer):
            return False
        message.content = "".join(buffer)
        buffer[:] = [message.content]
        self._stream_materialized_counts[message.id] = 1
        self._bump_message_speech_revision(message.id)
        return True

    def _materialize_stream_buffer(self, message: ConsoleChatMessage) -> None:
        """Fold buffered chunks and persist a newly materialized pending row."""
        if self._fold_stream_buffer_without_persistence(message):
            self._persist_pending_message_if_ready(message)

    @staticmethod
    def _snapshot(message: ConsoleChatMessage) -> ConsoleChatMessage:
        return replace(message)

    @staticmethod
    def _persistence_scope(session: ConsoleChatSession) -> tuple[str, str | None]:
        if session.workspace_id and session.workspace_id != CONSOLE_GLOBAL_WORKSPACE_ID:
            return "workspace", session.workspace_id
        return CONSOLE_GLOBAL_WORKSPACE_ID, None

    @staticmethod
    def _validate_can_stream(message: ConsoleChatMessage) -> None:
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can receive stream chunks.")
        if message.status not in {"pending", "streaming"}:
            raise ValueError(
                f"Cannot append stream chunks to a {message.status} message."
            )

    @staticmethod
    def _validate_can_mark_terminal(message: ConsoleChatMessage) -> None:
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError(
                "Only assistant messages can enter terminal stream states."
            )
        if message.status not in {"pending", "streaming"}:
            raise ValueError(f"Cannot mark a {message.status} message terminal.")

    @staticmethod
    def _initial_status(
        *,
        role: ConsoleMessageRole,
        content: str,
    ) -> ConsoleMessageStatus:
        if role is ConsoleMessageRole.ASSISTANT and not content:
            return "pending"
        return "complete"
