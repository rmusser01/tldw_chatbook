import base64
import json
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from loguru import logger as _logger

from tldw_chatbook.Chat.citation_trace_models import SealedCitationWrite
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
    CitationTraceRepository,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleContextRepository,
    ContextPolicyReadResult,
)
from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpoint,
    ConsoleDurableTurnAcceptance,
)
from tldw_chatbook.Chat.console_library_policy_repository import (
    ConsoleLibraryPolicyRepository,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicySnapshot,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleTransactionContribution,
    _scoped_console_transaction_writer,
)
from tldw_chatbook.Chat.console_prefill import PINNED_PREFILL_METADATA_KEY
from tldw_chatbook.Chat.console_roleplay_metadata import (
    ConsoleRoleplayContext,
    merge_console_roleplay_context,
    parse_console_roleplay_context,
)
from tldw_chatbook.Chat.console_speech_preferences import (
    ConsoleSpeechPreferences,
    merge_console_speech_preferences,
    parse_console_speech_preferences,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    TrajectoryRowWrite,
)

logger = _logger.bind(module="ChatPersistenceService")
_ASSISTANT_AUTHORITY_UNSET = cast(Optional[str], object())


def _initial_metadata_object(metadata: object) -> dict[str, object]:
    """Return strict JSON-object metadata without lossy key coercion."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} is not supported.")

    try:
        if isinstance(metadata, Mapping):
            candidate = dict(metadata)
            if not _mapping_keys_are_strings(candidate):
                raise ValueError("Mapping keys must be strings.")
            serialized = json.dumps(candidate, allow_nan=False, sort_keys=True)
            decoded = json.loads(serialized, parse_constant=reject_constant)
        elif type(metadata) is str:
            decoded = json.loads(metadata, parse_constant=reject_constant)
        else:
            raise ValueError("Unsupported metadata type.")
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("metadata must be a valid JSON object.") from exc
    if not isinstance(decoded, dict):
        raise ValueError("metadata must be a valid JSON object.")
    return decoded


def _mapping_keys_are_strings(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(
            type(key) is str and _mapping_keys_are_strings(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_mapping_keys_are_strings(item) for item in value)
    return True


class ChatPersistenceService:
    def __init__(
        self,
        db: CharactersRAGDB,
        workspace_registry: Any | None = None,
        citation_repository: CitationTraceRepository | None = None,
    ):
        self.db = db
        self.workspace_registry = workspace_registry
        self.citation_repository = citation_repository
        self.context_repository = ConsoleContextRepository(db)
        self.console_library_policy_repository = ConsoleLibraryPolicyRepository(db)
        self.console_dispatch_repository = ConsoleDispatchRepository(db)

    @staticmethod
    def thinking_round_trip_version() -> int:
        """Return the thinking envelope version this local adapter round-trips."""
        return 1

    @property
    def canonical_citation_writes_ready(self) -> bool:
        """Return whether this service can persist canonical local citations.

        Returns:
            True when the configured citation repository shares this service's
            database and is ready for local canonical writes.
        """

        repository = self.citation_repository
        return bool(
            repository is not None
            and repository.db is self.db
            and repository.local_citation_writes_ready
        )

    def get_message_version(self, message_id: str) -> int | None:
        """Return the current positive version for one non-deleted message.

        Args:
            message_id: Persisted Chat message identifier.

        Returns:
            The exact positive integer row version, or ``None`` when the row
            is missing, deleted, or carries an untrustworthy version value.
        """
        if type(message_id) is not str or not message_id:
            return None
        # TASK-22226: per-send settle/continuation reconciles call this for
        # just-written rows -- read the version without hydrating the BLOB.
        message = self.db.get_message_by_id_without_blob(message_id)
        if message is None or message.get("deleted"):
            return None
        version = message.get("version")
        if type(version) is not int or version < 1:
            return None
        return version

    def get_conversation_version(self, conversation_id: str) -> int | None:
        """Return the current positive version for one active conversation."""
        if not isinstance(conversation_id, str) or not conversation_id:
            return None
        conversation = self.db.get_conversation_by_id(conversation_id)
        if conversation is None or conversation.get("deleted"):
            return None
        version = conversation.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            return None
        return version

    def get_conversation_speech_preferences(
        self, conversation_id: str
    ) -> ConsoleSpeechPreferences:
        """Read fail-closed reply-speech preferences from conversation metadata."""
        if type(conversation_id) is not str or not conversation_id:
            return ConsoleSpeechPreferences()
        record = self.db.get_conversation_by_id(conversation_id)
        if record is None or record.get("deleted"):
            return ConsoleSpeechPreferences()
        return parse_console_speech_preferences(record.get("metadata"))

    def update_conversation_speech_preferences(
        self,
        *,
        conversation_id: str,
        preferences: ConsoleSpeechPreferences,
        expected_version: int,
    ) -> bool:
        """Merge speech metadata using the caller's exact conversation version."""
        if type(expected_version) is not int or expected_version < 1:
            return False
        record = self.db.get_conversation_by_id(str(conversation_id))
        if record is None or record.get("version") != expected_version:
            return False
        metadata = merge_console_speech_preferences(
            record.get("metadata"),
            preferences,
        )
        return bool(
            self.db.update_conversation(
                str(conversation_id),
                {"metadata": json.dumps(metadata, sort_keys=True)},
                expected_version=expected_version,
            )
        )

    def get_conversation_context_policy(
        self, conversation_id: str
    ) -> ContextPolicyReadResult:
        """Return local sparse context-policy overrides for one conversation."""
        return self.context_repository.load_policy(conversation_id)

    def update_conversation_context_policy(
        self,
        *,
        conversation_id: str,
        overrides: ConsoleContextPolicyOverrides,
    ) -> int | None:
        """Persist local sparse context-policy overrides without sync writes."""
        return self.context_repository.save_policy(conversation_id, overrides)

    @staticmethod
    def derive_conversation_title(
        *,
        character_name: Optional[str] = None,
        assistant_kind: Optional[str] = None,
        assistant_id: Optional[str] = None,
        explicit_title: Optional[str] = None,
    ) -> str:
        if explicit_title:
            return explicit_title
        if character_name:
            return f"Chat with {character_name}"

        normalized_kind = (assistant_kind or "").strip().lower() or None
        normalized_id = (assistant_id or "").strip() or None

        if normalized_kind == "persona":
            return (
                f"Chat with {normalized_id}" if normalized_id else "Chat with Persona"
            )
        if normalized_kind == "character":
            return (
                f"Chat with {normalized_id}" if normalized_id else "Chat with Character"
            )
        return "New Chat"

    def create_conversation(
        self,
        *,
        conversation_id: str | None = None,
        character_id: Optional[int] = None,
        character_name: Optional[str] = None,
        assistant_kind: Optional[str] = None,
        assistant_id: Optional[str] = None,
        assistant_authority_id: Optional[str] = _ASSISTANT_AUTHORITY_UNSET,
        persona_memory_mode: Optional[str] = None,
        runtime_backend: Optional[str] = None,
        discovery_owner: Optional[str] = None,
        discovery_entity_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        conversation_title: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Mapping[str, object] | str | None = None,
        speech_preferences: ConsoleSpeechPreferences | None = None,
        thinking_history_policy: str | None = None,
    ) -> str:
        """Create a conversation after validating any workspace authority.

        Args:
            conversation_id: Optional preallocated durable conversation identity.
            character_id: Local character identifier associated with the conversation.
            character_name: Display name used to derive a title when no explicit
                title is supplied.
            assistant_kind: Kind of assistant that owns the conversation.
            assistant_id: Stable assistant identifier used for title derivation.
            assistant_authority_id: Provenance authority identifier. Omitting it
                leaves the field absent so eligible DB-owned local inference may
                apply; passing ``None`` explicitly preserves unproven authority.
            persona_memory_mode: Memory behavior for a persona conversation.
            runtime_backend: Backend selected to run the assistant.
            discovery_owner: Owner of the assistant discovery record.
            discovery_entity_id: Discovery record identifier for the assistant.
            scope_type: Conversation scope. Only an explicit normalized
                ``scope_type="workspace"`` validates the workspace target here.
                Registry membership is a separate post-commit projection.
            workspace_id: Candidate workspace identifier forwarded to the
                database and resolved for an explicit workspace scope.
                Non-workspace/global persistence is normalized by the database
                and may clear it; this method never creates registry membership.
            conversation_title: Explicit title, which takes precedence when
                truthy; otherwise the character or assistant-derived title is used.
            system_prompt: Initial system prompt persisted with the conversation.
            metadata: Optional initial conversation metadata mapping or JSON object
                string. Malformed or non-object values are rejected before creation.
            speech_preferences: Optional staged Console reply-speech preferences
                to include in the conversation metadata before returning.
            thinking_history_policy: Optional normalized conversation replay
                preference. Null and missing values retain legacy Auto behavior.

        Returns:
            Persisted conversation ID.

        Raises:
            ValueError: If workspace scope is invalid or its workspace cannot be
                resolved. Workspace registry membership is intentionally not
                written here; it is a post-commit projection of the durable row.
        """
        safe_workspace_id = self._require_workspace_scope(
            scope_type=scope_type,
            workspace_id=workspace_id,
        )
        title = self.derive_conversation_title(
            character_name=character_name,
            assistant_kind=assistant_kind,
            assistant_id=assistant_id,
            explicit_title=conversation_title,
        )
        conversation_data = {
            "character_id": character_id,
            "assistant_kind": assistant_kind,
            "assistant_id": assistant_id,
            "persona_memory_mode": persona_memory_mode,
            "runtime_backend": runtime_backend,
            "discovery_owner": discovery_owner,
            "discovery_entity_id": discovery_entity_id,
            "scope_type": scope_type,
            "workspace_id": safe_workspace_id
            if safe_workspace_id is not None
            else workspace_id,
            "title": title,
            "system_prompt": system_prompt,
            "thinking_history_policy": thinking_history_policy,
            "client_id": self.db.client_id,
        }
        if conversation_id is not None:
            conversation_data["id"] = conversation_id
        if assistant_authority_id is not _ASSISTANT_AUTHORITY_UNSET:
            conversation_data["assistant_authority_id"] = assistant_authority_id
        initial_metadata = (
            _initial_metadata_object(metadata) if metadata is not None else None
        )
        if (
            speech_preferences is not None
            and speech_preferences != ConsoleSpeechPreferences()
        ):
            initial_metadata = merge_console_speech_preferences(
                initial_metadata,
                speech_preferences,
            )
        if initial_metadata is not None:
            conversation_data["metadata"] = json.dumps(
                initial_metadata,
                allow_nan=False,
                sort_keys=True,
            )
        return self.db.add_conversation(conversation_data)

    def persist_console_conversation_with_policy(
        self,
        *,
        conversation_id: str,
        policy_candidate: ConsoleLibraryPolicyCandidate,
        conversation_kwargs: Mapping[str, object],
    ) -> ConsoleLibraryPolicySnapshot:
        """Commit a first conversation row and its Library policy together."""
        self.validate_workspace_target(**conversation_kwargs)
        with self.db.transaction(immediate=True):
            created_id = self.create_conversation(
                conversation_id=conversation_id,
                **dict(conversation_kwargs),
            )
            if created_id != conversation_id:
                raise RuntimeError(
                    "Persistence returned an unexpected conversation id."
                )
            result = self.console_library_policy_repository.insert(
                conversation_id,
                policy_candidate,
            )
            if result.status is not ConsoleLibraryPolicyWriteStatus.COMMITTED:
                raise RuntimeError(
                    "Console Library policy could not be committed with conversation."
                )
        return result.snapshot

    def commit_durable_turn(
        self,
        *,
        acceptance: ConsoleDurableTurnAcceptance,
        policy_candidate: ConsoleLibraryPolicyCandidate,
        conversation_kwargs: Mapping[str, object],
    ) -> ConsoleDispatchCheckpoint:
        """Atomically create/validate and accept one durable Console turn.

        The service owns the sole outer ``BEGIN IMMEDIATE``.  It intentionally
        returns only durable values and never mutates the live Console session;
        publication is a postcommit store/controller responsibility.
        """

        self.validate_workspace_target(**conversation_kwargs)
        with self.db.transaction(immediate=True) as cursor:
            conversation = cursor.execute(
                "SELECT deleted FROM conversations WHERE id = ?",
                (acceptance.conversation_id,),
            ).fetchone()
            if conversation is None:
                created_id = self.create_conversation(
                    conversation_id=acceptance.conversation_id,
                    **dict(conversation_kwargs),
                )
                if created_id != acceptance.conversation_id:
                    raise RuntimeError(
                        "Persistence returned an unexpected conversation id."
                    )
                policy_result = self.console_library_policy_repository.insert(
                    acceptance.conversation_id,
                    policy_candidate,
                )
                if (
                    policy_result.status
                    is not ConsoleLibraryPolicyWriteStatus.COMMITTED
                ):
                    raise RuntimeError(
                        "Console Library policy could not be committed with turn."
                    )
            else:
                if conversation["deleted"]:
                    raise RuntimeError("Durable conversation is unavailable.")
                policy_row = cursor.execute(
                    "SELECT auto_retrieve_on_send, assistant_library_access, "
                    "policy_revision FROM console_conversation_library_policy "
                    "WHERE conversation_id = ?",
                    (acceptance.conversation_id,),
                ).fetchone()
                authority = acceptance.frozen_authority.policy
                if (
                    policy_row is None
                    or authority.source != "durable"
                    or authority.policy_revision != policy_row["policy_revision"]
                    or int(policy_candidate.auto_retrieve.value == "automatic")
                    != policy_row["auto_retrieve_on_send"]
                    or int(policy_candidate.assistant_access.value == "allowed")
                    != policy_row["assistant_library_access"]
                ):
                    raise RuntimeError(
                        "Durable Console Library policy no longer matches acceptance."
                    )
            return self.console_dispatch_repository.insert_with_messages(
                cursor,
                acceptance,
            )

    def promote_console_conversation_bundle(
        self,
        *,
        conversation_id: str,
        policy_candidate: ConsoleLibraryPolicyCandidate,
        conversation_kwargs: Mapping[str, object],
        messages: Sequence[Mapping[str, object]],
        active_leaf_message_id: str | None,
        context_summary: str | None = None,
        context_summary_boundary_message_id: str | None = None,
        contributions: Sequence[ConsoleTransactionContribution] = (),
    ) -> ConsoleLibraryPolicySnapshot:
        """Commit one temporary Console transcript and all Task-7 sidecars."""
        self.validate_workspace_target(**conversation_kwargs)
        with self.db.transaction(immediate=True) as cursor:
            created_id = self.create_conversation(
                conversation_id=conversation_id,
                **dict(conversation_kwargs),
            )
            if created_id != conversation_id:
                raise RuntimeError(
                    "Persistence returned an unexpected conversation id."
                )
            policy_result = self.console_library_policy_repository.insert(
                conversation_id,
                policy_candidate,
            )
            if policy_result.status is not ConsoleLibraryPolicyWriteStatus.COMMITTED:
                raise RuntimeError(
                    "Console Library policy could not be committed with conversation."
                )
            message_ids: dict[str, str] = {}
            for prepared in messages:
                native_id = str(prepared["native_id"])
                kwargs = dict(prepared["create_kwargs"])
                persisted_id = self.create_message(
                    conversation_id=conversation_id,
                    **kwargs,
                )
                expected_id = str(kwargs["message_id"])
                if persisted_id != expected_id:
                    raise RuntimeError("Persistence returned an unexpected message id.")
                message_ids[native_id] = persisted_id
                role = str(kwargs["sender"])
                if role in {"user", "assistant"}:
                    message_ids[role] = persisted_id
            self.db.set_conversation_active_leaf(
                conversation_id,
                active_leaf_message_id,
            )
            self.db.set_conversation_context_summary(
                conversation_id,
                context_summary,
                context_summary_boundary_message_id,
            )
            if contributions:
                with _scoped_console_transaction_writer(
                    cursor,
                    conversation_id,
                ) as writer:
                    for contribution in contributions:
                        contribution.write(
                            writer=writer,
                            conversation_id=conversation_id,
                            message_ids=message_ids,
                        )
        return policy_result.snapshot

    def fork_conversation_into_workspace(
        self,
        *,
        conversation_id: str,
        target_workspace_id: str,
    ) -> Any:
        """Record a workspace conversation link without mutating global history.

        Args:
            conversation_id: Existing conversation id to expose in the target
                workspace context.
            target_workspace_id: Workspace id that should receive the
                conversation membership link.

        Returns:
            The workspace membership returned by the registry service.

        Raises:
            ValueError: If the conversation or target workspace cannot be
                resolved.
            Exception: Propagates registry storage failures from the workspace
                membership link operation.
        """

        conversation = self.db.get_conversation_by_id(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        safe_workspace_id = self._require_workspace_scope(
            scope_type="workspace",
            workspace_id=target_workspace_id,
        )
        if safe_workspace_id is None:
            raise ValueError("Failed to resolve a valid workspace ID")
        title = str(conversation.get("title") or "Workspace conversation")
        return self._link_workspace_conversation(
            workspace_id=safe_workspace_id,
            conversation_id=conversation_id,
            title=title,
        )

    def _require_workspace_scope(
        self,
        *,
        scope_type: Optional[str],
        workspace_id: Optional[str],
    ) -> Optional[str]:
        normalized_scope = (scope_type or "").strip().lower()
        if normalized_scope != "workspace":
            return None
        safe_workspace_id = (workspace_id or "").strip()
        if not safe_workspace_id:
            raise ValueError("Workspace conversation requires a workspace_id")
        if self.workspace_registry is None:
            raise ValueError(
                "Workspace registry is required for workspace conversations"
            )
        workspace = self.workspace_registry.get_workspace(safe_workspace_id)
        if workspace is None:
            raise ValueError(f"Unknown workspace: {safe_workspace_id}")
        return safe_workspace_id

    def validate_workspace_target(self, **conversation_kwargs: object) -> str | None:
        """Validate an intended workspace before opening a Chat transaction."""
        return self._require_workspace_scope(
            scope_type=conversation_kwargs.get("scope_type"),
            workspace_id=conversation_kwargs.get("workspace_id"),
        )

    def project_workspace_membership(self, conversation_id: str) -> Any | None:
        """Project durable workspace authority into the registry idempotently."""
        conversation = self.db.get_conversation_by_id(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        safe_workspace_id = self._require_workspace_scope(
            scope_type=conversation.get("scope_type"),
            workspace_id=conversation.get("workspace_id"),
        )
        if safe_workspace_id is None:
            return None
        return self._link_workspace_conversation(
            workspace_id=safe_workspace_id,
            conversation_id=conversation_id,
            title=str(conversation.get("title") or "Workspace conversation"),
        )

    def _link_workspace_conversation(
        self,
        *,
        workspace_id: str,
        conversation_id: str,
        title: str,
    ) -> Any:
        return self.workspace_registry.link_membership(
            workspace_id,
            item_type="conversation",
            item_id=conversation_id,
            role="workspace-thread",
            title=title,
        )

    def update_conversation_system_prompt(
        self,
        *,
        conversation_id: str,
        system_prompt: Optional[str],
        expected_roleplay_context: ConsoleRoleplayContext | None = None,
        expected_system_prompts: tuple[str | None, ...] | None = None,
        allow_source_owned_repair: bool = False,
        expected_roleplay_version: int | None = None,
    ) -> bool:
        """Update the persisted system prompt for an existing conversation.

        Args:
            conversation_id: UUID of the conversation to update.
            system_prompt: New system prompt text, or ``None``/blank to clear it.

        Returns:
            True if the update was applied.

        Raises:
            ValueError: If the conversation cannot be found.
        """
        current_conversation = self.db.get_conversation_by_id(conversation_id)
        if not current_conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        if (
            expected_roleplay_version is not None
            and current_conversation.get("version") != expected_roleplay_version
        ):
            return False
        if (
            expected_roleplay_context is not None
            and parse_console_roleplay_context(current_conversation.get("metadata"))
            != expected_roleplay_context
        ):
            return False
        if (
            expected_system_prompts is not None
            and not allow_source_owned_repair
            and current_conversation.get("system_prompt") not in expected_system_prompts
        ):
            return False

        return bool(
            self.db.update_conversation(
                conversation_id,
                {"system_prompt": system_prompt},
                expected_version=current_conversation["version"],
            )
        )

    def update_conversation_roleplay_context(
        self,
        *,
        conversation_id: str,
        user_name_override: str | None,
        character_system_template: str | None,
        character_name_snapshot: str | None,
    ) -> bool:
        """Merge Console-owned roleplay identity context with one retry.

        Re-reading the conversation before each bounded optimistic attempt is
        essential: a concurrent metadata writer can add unrelated sibling
        keys after our first read. Merging only the fresh record preserves
        those keys while this method changes its owned context.

        Args:
            conversation_id: Durable conversation identifier.
            user_name_override: Optional saved user display-name override.
            character_system_template: Optional saved character prompt template.
            character_name_snapshot: Optional historical character display name.

        Returns:
            True when the roleplay context was persisted; False when the
            conversation no longer exists.
        """
        for attempt in range(2):
            record = self.db.get_conversation_by_id(str(conversation_id))
            if record is None:
                return False
            metadata = merge_console_roleplay_context(
                record.get("metadata"),
                ConsoleRoleplayContext(
                    user_name_override=user_name_override,
                    character_system_template=character_system_template,
                    character_name_snapshot=character_name_snapshot,
                ),
            )
            try:
                self.db.update_conversation(
                    str(conversation_id),
                    {"metadata": metadata},
                    expected_version=record["version"],
                )
                return True
            except ConflictError:
                if attempt == 1:
                    raise
        return False

    def update_conversation_pinned_prefill(
        self,
        *,
        conversation_id: str,
        pinned_prefill: str | None,
    ) -> bool:
        """Set or clear the pinned response prefill in conversation metadata.

        Merge-safe: re-parses the current ``metadata`` JSON and rewrites only
        its own key, preserving siblings such as ``active_dictionaries``
        (mirrors ``LocalChatDictionaryService._write_active_dictionaries``).
        Optimistic-lock conflicts (``ConflictError``) propagate to the caller.

        Returns:
            True when the write happened; False when the conversation does
            not exist.
        """
        record = self.db.get_conversation_by_id(str(conversation_id))
        if record is None:
            return False
        try:
            meta = json.loads(record.get("metadata") or "{}")
        except (TypeError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        if pinned_prefill:
            meta[PINNED_PREFILL_METADATA_KEY] = pinned_prefill
        else:
            meta.pop(PINNED_PREFILL_METADATA_KEY, None)
        self.db.update_conversation(
            str(conversation_id),
            {"metadata": json.dumps(meta)},
            expected_version=record["version"],
        )
        return True

    def update_conversation_title(
        self,
        *,
        conversation_id: str,
        title: str,
    ) -> bool:
        """Update the persisted title for an existing conversation.

        Args:
            conversation_id: UUID of the conversation to update.
            title: New conversation title (already validated non-blank).

        Returns:
            True if the update was applied.

        Raises:
            ValueError: If the conversation cannot be found.
        """
        current_conversation = self.db.get_conversation_by_id(conversation_id)
        if not current_conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        return bool(
            self.db.update_conversation(
                conversation_id,
                {"title": title},
                expected_version=current_conversation["version"],
            )
        )

    def get_conversation_console_project_context(
        self, *, conversation_id: str
    ) -> str | None:
        """Return a conversation's local-only Console project-context JSON.

        Args:
            conversation_id: Durable conversation identifier.

        Returns:
            Stored versioned JSON, or ``None`` when unset or unavailable.
        """
        return self.db.get_conversation_console_project_context(conversation_id)

    def set_conversation_console_project_context(
        self,
        *,
        conversation_id: str,
        project_context_json: str | None,
    ) -> None:
        """Write local-only Console project-context JSON without sync churn.

        Args:
            conversation_id: Durable conversation identifier.
            project_context_json: Versioned control-state JSON, or ``None`` to
                clear it.
        """
        self.db.set_conversation_console_project_context(
            conversation_id, project_context_json
        )

    def update_message_content(
        self,
        *,
        message_id: str,
        content: str,
        image_data: Optional[bytes],
        image_mime_type: Optional[str],
        parent_message_id: Optional[str] = None,
        feedback: Optional[str] = None,
        update_parent: bool = False,
        update_feedback: bool = False,
        attachments: Optional[Sequence[Mapping[str, Any]]] = None,
        usage_json: Optional[str] = None,
        metadata_json: Optional[str] = None,
        expected_roleplay_template_source: str | None = None,
        expected_message_contents: tuple[str, ...] | None = None,
        allow_source_owned_repair: bool = False,
        expected_roleplay_version: int | None = None,
        preserve_provider_continuation: bool = False,
        preserve_descendants: bool = False,
        clear_generation_provenance: bool = False,
    ) -> bool:
        """Update a message's content, optionally its parent/feedback, and its images.

        Two mutually exclusive contracts govern how images are updated:

        Legacy (``attachments=None``): ``image_data``/``image_mime_type`` are
        the sole source of the message's single legacy image. Passing
        ``image_data=None`` leaves any already-persisted image untouched
        (it does NOT clear it) -- callers may pass ``None`` simply because
        in-memory bytes were never rehydrated. Passing non-``None``
        ``image_data`` replaces the persisted image.

        Split addressing (``attachments`` is a sequence): an authoritative,
        full rewrite of every position. Position 0 (if present) replaces the
        legacy ``image_data``/``image_mime_type`` columns -- even when its
        ``data``/``mime_type`` are ``None``, since supplying ``attachments``
        at all means the caller intends to overwrite. Positions >= 1 replace
        the ``message_attachments`` table rows via
        ``CharactersRAGDB.set_message_attachments`` (an empty list clears any
        stale rows). The row update and the table rewrite happen inside one
        transaction so a table-write failure rolls back the row update too;
        conversely, if the row update itself does not succeed (returns a
        falsy result without raising), the table write is skipped entirely
        so attachments never drift out of sync with unrevised content.
        ``image_data``/``image_mime_type`` are ignored when ``attachments``
        is supplied.

        Args:
            message_id: UUID of the message to update.
            content: New message text content.
            image_data: Legacy single-image bytes; ignored when
                ``attachments`` is supplied. See the legacy contract above
                for how ``None`` is handled.
            image_mime_type: Legacy single-image MIME type; ignored when
                ``attachments`` is supplied.
            parent_message_id: New parent message id, applied only when
                ``update_parent`` is True.
            feedback: New feedback value, applied only when
                ``update_feedback`` is True.
            update_parent: Whether to update ``parent_message_id``.
            update_feedback: Whether to update ``feedback``.
            attachments: Optional full 0..N-1 position list of attachment
                rows (each a mapping with ``position``, ``data``,
                ``mime_type``, and optional ``display_name``). When
                supplied, this is the sole, authoritative source for both
                the legacy image columns (position 0) and the
                ``message_attachments`` table (positions >= 1). ``None``
                leaves all attachments/images untouched by this call except
                via the legacy ``image_data``/``image_mime_type`` kwargs.
            usage_json: Optional normalized provider-usage JSON (Console
                cost ticker). Only included in the row update when not
                ``None``, so a content-only update (no usage known yet,
                e.g. a mid-stream edit) never overwrites an already-persisted
                value with NULL.
            metadata_json: Optional structured message metadata JSON
                (task-2364). Follows the same only-when-supplied rule as
                ``usage_json``, for the same reason.
            preserve_descendants: Skip descendant tombstones when this
                update belongs to an authoritative bulk-history resave.

        Returns:
            True if the row update was applied; False if the underlying
            update reported failure without raising (attachments are left
            untouched in that case).

        Raises:
            ValueError: If the message cannot be found.
            ConflictError: If the message was concurrently modified or
                soft-deleted (optimistic-lock version mismatch).
        """
        current_message = self.db.get_message_by_id(message_id)
        if not current_message:
            raise ValueError(f"Message {message_id} not found")
        if (
            expected_roleplay_version is not None
            and current_message.get("version") != expected_roleplay_version
        ):
            return False
        if expected_roleplay_template_source is not None:
            current_metadata = MessageMetadata.from_json(
                current_message.get("metadata_json")
            )
            if (
                current_metadata is None
                or current_metadata.template_kind != "character_greeting"
                or current_metadata.template_source != expected_roleplay_template_source
            ):
                return False
        if (
            expected_message_contents is not None
            and not allow_source_owned_repair
            and current_message.get("content") not in expected_message_contents
        ):
            return False

        update_data: Dict[str, Any] = {"content": content}
        # Only include the image columns when new image bytes are supplied.
        # ``ChaChaNotes_DB.update_message`` treats an *included* ``image_data``
        # key of ``None`` as an explicit request to NULL both image columns,
        # but omitting the key entirely leaves any persisted image untouched.
        # Callers here (e.g. the Console store) may pass ``image_data=None``
        # simply because in-memory bytes were never rehydrated -- that must
        # not wipe an image that already exists in the database.
        #
        # ``attachments`` (split addressing): ``None`` means "leave
        # attachments untouched" -- the byte-identical #621/#628-era behavior
        # above. When a caller passes a full 0..N-1 position list, position 0
        # replaces the legacy image columns (even when ``None``, since an
        # explicit attachments list is an authoritative rewrite) and
        # positions >= 1 replace the ``message_attachments`` table rows.
        if attachments is not None:
            position_zero = next(
                (row for row in attachments if int(row["position"]) == 0), None
            )
            update_data["image_data"] = position_zero["data"] if position_zero else None
            update_data["image_mime_type"] = (
                position_zero["mime_type"] if position_zero else None
            )
        elif image_data is not None:
            update_data["image_data"] = image_data
            update_data["image_mime_type"] = image_mime_type
        if update_parent:
            update_data["parent_message_id"] = parent_message_id
        if update_feedback:
            update_data["feedback"] = feedback
        # Only include usage_json when the caller actually has a value to
        # write -- shared by all three ``self.db.update_message`` call sites
        # below via this single ``update_data`` dict. Omitting the key (not
        # writing ``None``) leaves an already-persisted usage value
        # untouched on a content-only update (e.g. a mid-stream edit before
        # usage is known).
        if usage_json is not None:
            update_data["usage_json"] = usage_json
        # Same only-when-supplied contract for the local-only
        # ``metadata_json`` column (task-2364).
        if metadata_json is not None:
            update_data["metadata_json"] = metadata_json
        if clear_generation_provenance:
            update_data["thinking_blocks_json"] = None

        citation_repository = self.citation_repository
        if citation_repository is not None and citation_repository.db is not self.db:
            raise CitationPersistenceUnavailable(
                "citation_repository_database_mismatch"
            )

        if attachments is not None:
            extra_rows = [
                {
                    "position": int(row["position"]),
                    "data": row["data"],
                    "mime_type": row["mime_type"],
                    "display_name": row.get("display_name", ""),
                }
                for row in attachments
                if int(row["position"]) >= 1
            ]
        else:
            extra_rows = []

        if citation_repository is not None:
            # IMMEDIATE (task-21100): `transaction(immediate=...)` is honored
            # only at depth 0, so this OUTER wrapper decides the begin mode for
            # the nested hot messages writers -- left DEFERRED it re-opens the
            # snapshot-upgrade "database is locked" window their own IMMEDIATE
            # closes (see add_message's scoping comment).
            with self.db.transaction(immediate=True) as cursor:
                result = bool(
                    self.db.update_message(
                        message_id,
                        update_data,
                        expected_version=current_message["version"],
                        preserve_provider_continuation=preserve_provider_continuation,
                        preserve_descendants=preserve_descendants,
                    )
                )
                if result and attachments is not None:
                    self.db.set_message_attachments(message_id, extra_rows)
                if result:
                    citation_repository.transition_owner_for_message_update(
                        cursor,
                        message_id=message_id,
                        previous_revision=current_message["version"],
                        new_revision=current_message["version"] + 1,
                        new_body=content,
                    )
            return result

        if attachments is not None:
            # One atomic unit: inside this outer transaction the nested
            # update_message/set_message_attachments transactions are no-ops,
            # so a failed table write rolls back the row update (content and
            # legacy image columns) too. Conversely, if the row update itself
            # fails without raising (e.g. an optimistic-lock miss reported as
            # a plain ``False`` return instead of a ``ConflictError``), the
            # attachments table write must be skipped -- otherwise
            # attachments would be rewritten while content/version were not,
            # leaving the two out of sync.
            # IMMEDIATE (task-21100): outer wrappers decide the begin mode for
            # nested writers (immediate= is depth-0 only); DEFERRED here
            # re-opens the snapshot-upgrade window (see add_message).
            with self.db.transaction(immediate=True):
                result = bool(
                    self.db.update_message(
                        message_id,
                        update_data,
                        expected_version=current_message["version"],
                        preserve_provider_continuation=preserve_provider_continuation,
                        preserve_descendants=preserve_descendants,
                    )
                )
                if result:
                    self.db.set_message_attachments(message_id, extra_rows)
            return result

        return bool(
            self.db.update_message(
                message_id,
                update_data,
                expected_version=current_message["version"],
                preserve_provider_continuation=preserve_provider_continuation,
                preserve_descendants=preserve_descendants,
            )
        )

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
        """Replace and return one committed selected-generation version."""
        return self.db.replace_assistant_generation_projection(
            message_id=message_id,
            content=content,
            thinking_blocks_json=thinking_blocks_json,
            provider_continuation_json=provider_continuation_json,
            assistant_generation_state=assistant_generation_state,
            usage_json=usage_json,
            expected_version=expected_version,
        )

    def update_message_usage(self, *, message_id: str, usage_json: str) -> bool:
        """Persist a message's normalized usage WITHOUT touching sync metadata.

        Routes to :meth:`CharactersRAGDB.update_message_usage_local`, a
        direct, version-neutral column write, rather than
        ``update_message_content``/``self.db.update_message``. The Console
        cost ticker's ``usage_json`` column is local-only (derived from this
        device's own provider responses, never part of the sync payload), so
        going through the general-purpose row updater -- which always bumps
        ``version``/``last_modified`` -- would trip the ``messages_sync_update``
        trigger's ``WHEN`` clause on those two columns alone and enqueue a
        ``sync_log`` row whose payload can never carry the ``usage_json``
        that actually changed: pure cross-device churn (and a spurious
        optimistic-lock version bump) for a write with no syncable content.

        Callers should use this ONLY for a usage-only write (e.g. the Console
        store's stop-path terminal flush, attaching usage after the message's
        content/version have already been persisted). A normal write where
        usage rides alongside changed content still belongs on
        ``update_message_content``, since content changing a version IS a
        legitimate, syncable change.

        Args:
            message_id: UUID of the message to update.
            usage_json: Normalized ``ProviderUsage.to_json()`` payload.

        Returns:
            True if a non-deleted message with this id was found and
            updated; False otherwise.
        """
        return self.db.update_message_usage_local(message_id, usage_json)

    def append_message_exchanges(
        self, *, message_id: str, rows: Sequence[Mapping[str, Any]]
    ) -> bool:
        """Local-only exchange-capture flush (Conversation Inspector).

        Same contract as ``update_message_usage``: version-neutral, never
        enqueues sync rows. Unlike that sibling, this never lets a database
        error escape -- exchange captures are best-effort diagnostic
        payloads, not user-visible content, so a write failure is logged
        under the stable ``exchange_append_failed`` category with only
        ``message_id`` and the exception type -- never exception text, row
        contents, or capture payloads -- and reported as ``False`` rather
        than propagated.

        Args:
            message_id: UUID of the owning message row.
            rows: Exchange rows to upsert; see
                :meth:`CharactersRAGDB.append_message_exchanges_local`, including
                local-only capture provenance.

        Returns:
            True if the rows were written; False if the write failed.
        """
        try:
            self.db.append_message_exchanges_local(message_id, rows)
            return True
        except Exception as exc:  # noqa: BLE001 -- best-effort capture flush
            logger.bind(message_id=message_id, error_type=type(exc).__name__).warning(
                "exchange_append_failed"
            )
            return False

    def list_full_exchange_keys_for_conversation(
        self, conversation_id: str
    ) -> set[tuple[str, str, int]]:
        """Return queryable Full exchange keys for one conversation."""
        return self.db.list_full_exchange_keys_for_conversation(conversation_id)

    def delete_full_exchanges_for_conversation(
        self,
        conversation_id: str,
        *,
        expected_count: int | None = None,
    ) -> int:
        """Delete only Full exchange rows for one conversation."""
        return self.db.delete_full_exchanges_for_conversation(
            conversation_id,
            expected_count=expected_count,
        )

    def delete_message_subtree(self, *, message_id: str) -> list[dict[str, Any]]:
        """Atomically tombstone one persisted branch and return its versions."""
        current_message = self.db.get_message_by_id(message_id)
        if not current_message:
            raise ValueError(f"Message {message_id} not found")
        return self.db.soft_delete_message_subtree(
            message_id,
            expected_version=current_message["version"],
        )

    def write_trajectory_rows(self, rows: Sequence[TrajectoryRowWrite]) -> bool:
        """Persist trajectory sidecar rows; LOCAL-ONLY, never raises.

        The trajectory sibling of :meth:`update_message_usage`: the
        ``message_trajectory_metadata`` sidecar (schema v38) is local-only
        with no sync triggers, so it never routes through the
        version-bumping general-purpose row updater. A small bounded retry
        absorbs transient write-write lock contention (concurrent Console
        sessions, compaction auxiliary turns): ``upsert_trajectory_rows``
        assigns ``seq`` inside its own transaction, so a rolled-back
        attempt simply re-derives seqs on retry. Returns ``False`` (after
        logging with row COUNT only -- never message contents or payloads)
        when every attempt failed, so the store's best-effort capture knows
        the batch was dropped.

        Args:
            rows: Sidecar rows to upsert.

        Returns:
            True when the rows were written; False when all attempts failed.
        """
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                self.db.upsert_trajectory_rows(rows)
                return True
            except Exception as exc:  # noqa: BLE001 -- never fail the turn
                last_error = exc
                # Brief escalating backoff: the losing writer of a
                # concurrent pair only needs the winner's commit to land.
                time.sleep(0.02 * (attempt + 1))
        logger.bind(row_count=len(rows), error=repr(last_error)).warning(
            "trajectory_rows_write_failed"
        )
        return False

    def update_message_metadata(self, *, message_id: str, metadata_json: str) -> bool:
        """Persist structured message metadata WITHOUT touching sync metadata.

        The metadata sibling of :meth:`update_message_usage`, routed to
        :meth:`CharactersRAGDB.update_message_metadata_local` for the same
        reason: ``metadata_json`` (task-2364) is local-only, so a
        metadata-only write through the general-purpose row updater would
        bump ``version``/``last_modified``, trip the
        ``messages_sync_update`` trigger and enqueue a ``sync_log`` row
        whose payload can never carry the column that changed.

        Use this ONLY for a metadata-only write against an already-persisted
        row (e.g. marking a reply interrupted after it was flushed). When
        metadata rides changed content, ``update_message_content``'s
        ``metadata_json`` kwarg is the right seam -- the version bump there
        belongs to the content.

        Args:
            message_id: UUID of the message to update.
            metadata_json: ``MessageMetadata.to_json()`` payload.

        Returns:
            True if a non-deleted message with this id was found and
            updated; False otherwise.
        """
        return self.db.update_message_metadata_local(message_id, metadata_json)

    def create_message(
        self,
        *,
        conversation_id: str,
        sender: str,
        content: str,
        image_data: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        message_id: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        feedback: Optional[str] = None,
        attachments: Optional[Sequence[Mapping[str, Any]]] = None,
        generation_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        citation_write: SealedCitationWrite | None = None,
        usage_json: Optional[str] = None,
        metadata_json: Optional[str] = None,
        thinking_blocks_json: Optional[str] = None,
        provider_continuation_json: Optional[str] = None,
        assistant_generation_state: Optional[str] = None,
    ) -> str:
        """Create a new message, optionally with a legacy image or a full attachment list.

        Two mutually exclusive contracts govern how images are stored,
        mirroring ``update_message_content``:

        Legacy (``attachments=None``): ``image_data``/``image_mime_type``
        are the sole source of the message's single legacy image, stored
        directly on the ``messages`` row. The ``message_attachments`` table
        is never touched.

        Split addressing (``attachments`` is a sequence): an authoritative
        full 0..N-1 position list. Position 0 (if present) overrides the
        scalar ``image_data``/``image_mime_type`` kwargs -- even overriding
        them with ``None`` when no position-0 entry is present, since
        supplying ``attachments`` at all means it is authoritative. Positions
        >= 1 are written to the ``message_attachments`` table via
        ``CharactersRAGDB.set_message_attachments``, always -- even an empty
        list, so any stale rows a prior attempt at this same ``message_id``
        left behind are cleared. The row insert and the table write happen
        inside one transaction, so a table-write failure rolls back the row
        insert too.

        Args:
            conversation_id: UUID of the parent conversation.
            sender: Message sender/role (e.g. ``"user"``, ``"assistant"``).
            content: Message text content.
            image_data: Legacy single-image bytes; ignored when
                ``attachments`` is supplied.
            image_mime_type: Legacy single-image MIME type; ignored when
                ``attachments`` is supplied.
            message_id: Optional explicit message id; the DB generates one
                when omitted.
            parent_message_id: Optional parent message id for threading.
            feedback: Optional feedback value applied via a follow-up update
                once the message exists (feedback is not part of the initial
                insert payload).
            attachments: Optional full 0..N-1 position list of attachment
                rows (each a mapping with ``position``, ``data``,
                ``mime_type``, and optional ``display_name``). When
                supplied, this is the sole, authoritative source for both
                the legacy image columns (position 0) and the
                ``message_attachments`` table (positions >= 1).
            generation_metadata: Optional full list of
                ``message_generation_metadata`` rows (each a mapping with
                ``position``, ``prompt``, ``negative_prompt``, ``backend``,
                ``model``, ``seed``, ``style``, ``params_json``) to persist
                alongside the message. Written via
                ``CharactersRAGDB.set_message_generation_metadata`` inside
                the same transaction as the row insert and the attachments
                write, so a sidecar-write failure rolls back everything
                (including the message row and any attachments already
                written this call).
            citation_write: Optional complete sealed citation aggregate.
                When present, it is preflighted before the transaction and
                committed atomically with the message, attachments,
                generation metadata, and feedback.
            usage_json: Optional normalized provider-usage JSON (Console
                cost ticker), written into the row's local-only
                ``usage_json`` column via ``CharactersRAGDB.add_message``.
            metadata_json: Optional structured message metadata JSON
                (task-2364: engine provenance, interrupted flag, transcript
                status), written into the row's local-only
                ``metadata_json`` column via ``CharactersRAGDB.add_message``.
            thinking_blocks_json: Canonical thinking envelope owned by this
                initial assistant generation.
            provider_continuation_json: Canonical private continuation owned
                by this initial assistant generation.
            assistant_generation_state: Portable lifecycle state for the
                initial assistant generation.

        Returns:
            The newly created message's id.

        Raises:
            CitationPersistenceUnavailable: If citation persistence is
                disabled, misconfigured, invalid, or bound to another DB.
            CharactersRAGDBError: For database integrity errors during the
                row insert -- ``CharactersRAGDB.add_message`` wraps
                ``sqlite3.IntegrityError`` into this explicitly. The
                attachment-table and generation-metadata sidecar writes
                (``set_message_attachments``/``set_message_generation_metadata``)
                run through the plain transaction cursor with no
                independent wrap, matching ``update_message_content``'s
                sibling contract for the identical ``set_message_attachments``
                call -- a raw ``sqlite3.Error`` can propagate from those two
                writes instead.
        """
        prepared_citation = None
        if citation_write is not None:
            if self.citation_repository is None:
                raise CitationPersistenceUnavailable("citation_repository_unavailable")
            if self.citation_repository.db is not self.db:
                raise CitationPersistenceUnavailable(
                    "citation_repository_database_mismatch"
                )
            prepared_citation = self.citation_repository.prepare_write(citation_write)

        # Split addressing: when ``attachments`` is supplied it covers ALL
        # positions (0..N-1) and is authoritative -- position 0 overrides the
        # scalar ``image_data``/``image_mime_type`` kwargs (even overriding
        # them with ``None`` when no position-0 entry is present), and
        # positions >= 1 land in the ``message_attachments`` table via
        # ``set_message_attachments``. ``attachments=None`` leaves the
        # scalar kwargs as the sole source of the legacy image columns and
        # never touches the attachments table -- byte-identical to the
        # pre-split behavior.
        effective_image_data = image_data
        effective_image_mime_type = image_mime_type
        extra_rows: List[Dict[str, Any]] = []
        if attachments is not None:
            position_zero = next(
                (row for row in attachments if int(row["position"]) == 0), None
            )
            effective_image_data = position_zero["data"] if position_zero else None
            effective_image_mime_type = (
                position_zero["mime_type"] if position_zero else None
            )
            extra_rows = [
                {
                    "position": int(row["position"]),
                    "data": row["data"],
                    "mime_type": row["mime_type"],
                    "display_name": row.get("display_name", ""),
                }
                for row in attachments
                if int(row["position"]) >= 1
            ]

        message_payload = {
            "id": message_id,
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "sender": sender,
            "content": content,
            "image_data": effective_image_data,
            "image_mime_type": effective_image_mime_type,
            "client_id": self.db.client_id,
            "usage_json": usage_json,
            "metadata_json": metadata_json,
            "thinking_blocks_json": thinking_blocks_json,
            "provider_continuation_json": provider_continuation_json,
            "assistant_generation_state": assistant_generation_state,
        }
        if prepared_citation is not None:
            # IMMEDIATE (task-21100): outer wrappers decide the begin mode for
            # nested writers (immediate= is depth-0 only); DEFERRED here
            # re-opens the snapshot-upgrade window (see add_message).
            with self.db.transaction(immediate=True) as cursor:
                existing_message = (
                    self.db.get_message_by_id(message_id)
                    if message_id is not None
                    else None
                )
                if existing_message is not None:
                    self._verify_citation_message_retry(
                        existing_message=existing_message,
                        message_payload=message_payload,
                        feedback=feedback,
                        extra_rows=extra_rows,
                        generation_metadata=generation_metadata,
                    )
                    created_message_id = existing_message["id"]
                else:
                    created_message_id = self.db.add_message(message_payload)
                    if attachments is not None:
                        self.db.set_message_attachments(created_message_id, extra_rows)
                    if generation_metadata is not None:
                        self.db.set_message_generation_metadata(
                            created_message_id, list(generation_metadata)
                        )
                    if feedback is not None:
                        # TASK-22226: this readback only feeds the feedback
                        # update's optimistic lock -- read the DB-normalized
                        # version without hydrating the image BLOB.
                        created_message = self.db.get_message_by_id_without_blob(
                            created_message_id
                        )
                        self.db.update_message(
                            created_message_id,
                            {"feedback": feedback},
                            expected_version=created_message["version"],
                        )
                # TASK-22226: only ``version`` (message_revision) is consumed
                # here; write_prepared independently re-validates it against
                # the messages row inside this same transaction.
                created_message = self.db.get_message_by_id_without_blob(
                    created_message_id
                )
                self.citation_repository.write_prepared(
                    cursor,
                    prepared_citation,
                    message_id=created_message_id,
                    message_revision=created_message["version"],
                    message_body=content,
                )
            return created_message_id
        if attachments is not None or generation_metadata is not None:
            # One atomic unit: inside this outer transaction the nested
            # add_message/set_message_attachments/set_message_generation_metadata
            # transactions are no-ops, so a failed table write rolls the
            # message row (and any earlier write in this call) back too. The
            # attachments write always runs when this branch is taken -- an
            # empty list still clears any stale rows a prior attempt at this
            # same message_id may have left behind.
            # IMMEDIATE (task-21100): outer wrappers decide the begin mode for
            # nested writers (immediate= is depth-0 only); DEFERRED here
            # re-opens the snapshot-upgrade window (see add_message).
            with self.db.transaction(immediate=True):
                created_message_id = self.db.add_message(message_payload)
                self.db.set_message_attachments(created_message_id, extra_rows)
                if generation_metadata is not None:
                    self.db.set_message_generation_metadata(
                        created_message_id, list(generation_metadata)
                    )
        else:
            created_message_id = self.db.add_message(message_payload)
        if feedback is not None:
            # TASK-22226: version-only readback -- never rehydrate the BLOB
            # that was just written.
            created_message = self.db.get_message_by_id_without_blob(
                created_message_id
            )
            self.db.update_message(
                created_message_id,
                {"feedback": feedback},
                expected_version=created_message["version"],
            )
        return created_message_id

    def _verify_citation_message_retry(
        self,
        *,
        existing_message: Mapping[str, Any],
        message_payload: Mapping[str, Any],
        feedback: str | None,
        extra_rows: Sequence[Mapping[str, Any]],
        generation_metadata: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        """Fail closed unless an uncertain retry targets the exact message."""

        expected_fields = {
            "id": message_payload["id"],
            "conversation_id": message_payload["conversation_id"],
            "parent_message_id": message_payload["parent_message_id"],
            "sender": message_payload["sender"],
            "content": message_payload["content"],
            "image_data": message_payload["image_data"],
            "image_mime_type": message_payload["image_mime_type"],
            "client_id": message_payload["client_id"],
            "usage_json": message_payload["usage_json"],
            "metadata_json": message_payload["metadata_json"],
            "provider_continuation_json": message_payload["provider_continuation_json"],
            "thinking_blocks_json": message_payload["thinking_blocks_json"],
            "assistant_generation_state": message_payload["assistant_generation_state"],
            "feedback": feedback,
        }
        if any(
            existing_message.get(field) != expected
            for field, expected in expected_fields.items()
        ):
            raise CitationPersistenceUnavailable("message_identity_conflict")
        existing_rows = self.db.get_attachments_for_messages(
            [existing_message["id"]]
        ).get(existing_message["id"], [])

        def attachment_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
            return (
                int(row["position"]),
                row["data"],
                row["mime_type"],
                row.get("display_name", ""),
            )

        if tuple(map(attachment_identity, existing_rows)) != tuple(
            map(attachment_identity, extra_rows)
        ):
            raise CitationPersistenceUnavailable("message_identity_conflict")

        existing_generation_metadata = self.db.get_generation_metadata_for_messages(
            [existing_message["id"]]
        ).get(existing_message["id"], [])

        def generation_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
            return (
                int(row["position"]),
                row["prompt"],
                row.get("negative_prompt", ""),
                row["backend"],
                row.get("model"),
                row.get("seed"),
                row.get("style"),
                row.get("params_json", "{}"),
            )

        if tuple(map(generation_identity, existing_generation_metadata)) != tuple(
            map(generation_identity, generation_metadata or ())
        ):
            raise CitationPersistenceUnavailable("message_identity_conflict")

    def append_message_attachment(
        self,
        message_id: str,
        *,
        data: bytes,
        mime_type: str,
        display_name: str = "",
        generation_metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Append one new image variant to a message, in place.

        Thin passthrough to
        ``CharactersRAGDB.append_message_attachment_with_metadata`` -- the
        narrow, additive counterpart to the full-list
        ``update_message_content(attachments=...)`` rewrite. Use this when a
        new variant (e.g. a regenerated image) should be added without
        risking any existing attachment's bytes.

        Args:
            message_id: Target message id; must already have a position-0
                image.
            data: The new variant's image bytes.
            mime_type: The new variant's MIME type.
            display_name: Optional label for the new variant.
            generation_metadata: Optional generation-metadata fields for the
                new position.

        Returns:
            The position assigned to the new variant (>= 1).

        Raises:
            ValueError: If the message does not exist or has no position-0
                image.
        """
        return self.db.append_message_attachment_with_metadata(
            message_id,
            data=data,
            mime_type=mime_type,
            display_name=display_name,
            generation_metadata=generation_metadata,
        )

    def keep_message_attachment(self, message_id: str, position: int) -> None:
        """Promote a stored variant to be the message's canonical image.

        Thin passthrough to
        ``CharactersRAGDB.swap_message_attachment_with_scalar``. Swaps the
        variant at ``position`` with the message's current position-0 image,
        byte-identical, touching only those two variants.

        Args:
            message_id: Target message id.
            position: The attachment position (>= 1) to promote.

        Raises:
            ValueError: If ``position < 1`` or no attachment exists there.
        """
        self.db.swap_message_attachment_with_scalar(message_id, position)

    def get_attachments_for_messages(
        self, message_ids: Sequence[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Batch-fetch extra (position >= 1) attachments for messages.

        Passthrough to ``CharactersRAGDB.get_attachments_for_messages``.
        Legacy position-0 images are not included here -- they live on the
        ``messages`` row itself (``image_data``/``image_mime_type``).

        Args:
            message_ids: Message ids to fetch attachment rows for.

        Returns:
            A mapping of message id to its list of attachment row dicts
            (each with ``position``, ``data``, ``mime_type``,
            ``display_name``), ordered by position. Message ids with no
            extra (position >= 1) attachments are omitted from the result.
        """
        return self.db.get_attachments_for_messages(message_ids)

    def get_generation_metadata_for_messages(
        self, message_ids: Sequence[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Batch-fetch generation-metadata sidecar rows for messages.

        Passthrough to ``CharactersRAGDB.get_generation_metadata_for_messages``
        -- feeds ``ConsoleChatStore.hydrate_generation_metadata`` at
        conversation load (P2a).

        Args:
            message_ids: Message ids to fetch generation-metadata rows for.

        Returns:
            A mapping of message id to its position-ordered
            generation-metadata row dicts; message ids with no sidecar rows
            are omitted.
        """
        return self.db.get_generation_metadata_for_messages(message_ids)

    def save_history(
        self,
        *,
        conversation_id: str,
        chatbot_history: List[Dict[str, Any]],
    ) -> int:
        existing_messages = self.db.get_messages_for_conversation(
            conversation_id,
            limit=10000,
            order_by_timestamp="ASC",
        )
        existing_by_id = {message["id"]: message for message in existing_messages}
        existing_by_position = [
            message
            for message in existing_messages
            if message.get("variant_of") is None
        ]
        consumed_existing_ids = set()

        saved_count = 0
        fallback_index = 0

        for message_obj in chatbot_history:
            sender = message_obj.get("role")
            if not sender or sender == "system":
                continue

            content, image_data, image_mime_type = self._extract_message_payload(
                message_obj
            )
            if not content and not image_data:
                continue

            message_id = message_obj.get("id")
            parent_message_id = message_obj.get("parent_message_id")
            feedback = message_obj.get("feedback")

            if message_id and message_id in existing_by_id:
                self.update_message_content(
                    message_id=message_id,
                    content=content,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    parent_message_id=parent_message_id,
                    feedback=feedback,
                    update_parent="parent_message_id" in message_obj,
                    update_feedback="feedback" in message_obj,
                    preserve_descendants=True,
                )
                consumed_existing_ids.add(message_id)
            elif message_id:
                self.create_message(
                    conversation_id=conversation_id,
                    sender=sender,
                    content=content,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    message_id=message_id,
                    parent_message_id=parent_message_id,
                    feedback=feedback,
                )
            else:
                while (
                    fallback_index < len(existing_by_position)
                    and existing_by_position[fallback_index]["id"]
                    in consumed_existing_ids
                ):
                    fallback_index += 1

            if not message_id and fallback_index < len(existing_by_position):
                existing_message = existing_by_position[fallback_index]
                self.update_message_content(
                    message_id=existing_message["id"],
                    content=content,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    parent_message_id=parent_message_id,
                    feedback=feedback,
                    update_parent="parent_message_id" in message_obj,
                    update_feedback="feedback" in message_obj,
                    preserve_descendants=True,
                )
                consumed_existing_ids.add(existing_message["id"])
                fallback_index += 1
            elif not message_id:
                self.create_message(
                    conversation_id=conversation_id,
                    sender=sender,
                    content=content,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    parent_message_id=parent_message_id,
                    feedback=feedback,
                )

            saved_count += 1

        retained_existing_ids = set(consumed_existing_ids)
        variants_added = True
        while variants_added:
            variants_added = False
            for existing_message in existing_messages:
                if existing_message["id"] in retained_existing_ids:
                    continue
                if existing_message.get("variant_of") in retained_existing_ids:
                    retained_existing_ids.add(existing_message["id"])
                    variants_added = True

        for existing_message in existing_messages:
            if existing_message["id"] not in retained_existing_ids:
                self.db.soft_delete_message(
                    existing_message["id"],
                    existing_message["version"],
                )

        return saved_count

    @staticmethod
    def _extract_message_payload(
        message_obj: Dict[str, Any],
    ) -> Tuple[str, Optional[bytes], Optional[str]]:
        text_content_parts: List[str] = []
        image_data_bytes: Optional[bytes] = None
        image_mime_type_str: Optional[str] = None
        content_data = message_obj.get("content")

        if isinstance(content_data, str):
            text_content_parts.append(content_data)
        elif isinstance(content_data, list):
            for part in content_data:
                part_type = part.get("type")
                if part_type == "text":
                    text_content_parts.append(part.get("text", ""))
                elif part_type == "image_url":
                    image_url_dict = part.get("image_url", {})
                    url_str = image_url_dict.get("url", "")
                    if url_str.startswith("data:") and ";base64," in url_str:
                        try:
                            header, b64_data = url_str.split(";base64,", 1)
                            image_mime_type_str = (
                                header.split("data:", 1)[1]
                                if "data:" in header
                                else None
                            )
                            if image_mime_type_str:
                                image_data_bytes = base64.b64decode(b64_data)
                            else:
                                text_content_parts.append(
                                    "<Error: Malformed image data URI in history>"
                                )
                        except Exception:
                            image_data_bytes = None
                            image_mime_type_str = None
                            text_content_parts.append(
                                "<Error: Failed to decode image data from history>"
                            )
                    elif url_str:
                        text_content_parts.append(f"<Image URL: {url_str}>")
        elif content_data is not None:
            text_content_parts.append(
                f"<Unsupported content type: {type(content_data)}>"
            )

        return (
            "\n".join(text_content_parts).strip(),
            image_data_bytes,
            image_mime_type_str,
        )
