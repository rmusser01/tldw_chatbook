import base64
import json
import time
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from loguru import logger as _logger

from tldw_chatbook.Chat.attachment_core import MAX_ATTACHMENT_BYTES
from tldw_chatbook.Chat.citation_trace_models import SealedCitationWrite
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
    CitationTraceRepository,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_chat_fork import (
    CONSOLE_FORK_VIDEO_TOMBSTONE_CONTENT,
    ConsoleChatForkSnapshot,
    encode_console_fork_message_metadata,
    fingerprint_console_fork_selected_image,
    validate_console_fork_image_payload,
)
from tldw_chatbook.Chat.console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleContextRepository,
    ContextPolicyReadResult,
    ContextPolicyWriteResult,
    ContextPolicyWriteStatus,
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
from tldw_chatbook.Chat.console_generation_settings_metadata import (
    ConsoleGenerationSettingsReadResult,
    ConsoleGenerationSettingsReadStatus,
    ConsoleGenerationSettingsSnapshot,
    ConsoleGenerationSettingsWriteResult,
    ConsoleGenerationSettingsWriteStatus,
    merge_console_generation_settings,
    parse_console_generation_settings,
    snapshot_from_session_settings,
    strict_json_metadata_object,
)
from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleTransactionContribution,
    _scoped_console_transaction_writer,
)
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    TraceForkBoundary,
)
from tldw_chatbook.Chat.console_semantic_revision import SemanticRevisionCoordinator
from tldw_chatbook.Chat.library_activity import LibraryActivityContribution
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
from tldw_chatbook.Chat.console_session_endpoint_policy import (
    ConsoleEndpointAdoptionReceipt,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_project_instructions import (
    encode_project_context_json,
)
from tldw_chatbook.Chat.rag_scope import serialize_scope
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    TrajectoryRowWrite,
)
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_trace_maintenance import TraceGCResult

logger = _logger.bind(module="ChatPersistenceService")
_ASSISTANT_AUTHORITY_UNSET = cast(Optional[str], object())
_CONTEXT_POLICY_EXPECTED_REVISION_UNSET = object()
CONSOLE_FORK_SOURCE_LINEAGE_MAX_DEPTH = 10_000


@dataclass(frozen=True, slots=True)
class ConsoleForkCommitResult:
    """Durable identities returned only after an atomic fork commit."""

    conversation_id: str
    active_leaf_message_id: str
    message_id_map: dict[str, str]
    policy: ConsoleLibraryPolicySnapshot
    already_committed: bool = False


def _initial_metadata_object(metadata: object) -> dict[str, object]:
    """Return strict JSON-object metadata without lossy key coercion."""
    return strict_json_metadata_object(metadata)


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
        self._console_trace_repository = ConsoleTraceRepository()
        self._semantic_revision_coordinator = SemanticRevisionCoordinator(
            db,
            repository=self._console_trace_repository,
        )

    @property
    def console_trace_repository(self) -> ConsoleTraceRepository:
        """Return the cursor-only semantic trace transaction participant."""
        return self._console_trace_repository

    @property
    def semantic_revision_coordinator(self) -> SemanticRevisionCoordinator:
        """Return the shared caller-transaction semantic mutation coordinator."""

        return self._semantic_revision_coordinator

    def purge_console_trace(
        self,
        *,
        conversation_id: str,
        request_id: str,
        detached_at: str,
    ) -> "TraceGCResult":
        """Detach and logically reclaim one conversation's unreachable trace."""

        from tldw_chatbook.Chat.console_trace_maintenance import TraceGarbageCollector

        return TraceGarbageCollector(self.db).purge_conversation(
            conversation_id=conversation_id,
            request_id=request_id,
            detached_at=detached_at,
        )

    def get_console_trace_fork_boundary(
        self,
        *,
        conversation_id: str,
        included_turn_ids: Sequence[str],
    ) -> TraceForkBoundary | None:
        """Read one immutable trace prefix boundary for a Console fork fence.

        Args:
            conversation_id: Durable source conversation identity.
            included_turn_ids: Ordered unique trace-turn identities in the forked
                message prefix.

        Returns:
            The newest reachable trace boundary for the prefix, or None when the
            source has no attached trace owner or matching events.
        """

        with self.db.transaction() as cursor:
            return self._console_trace_repository.capture_fork_boundary(
                cursor,
                conversation_id=conversation_id,
                included_turn_ids=included_turn_ids,
            )

    def settle_provider_response_trace(
        self,
        *,
        coordinator: object,
        request: object,
        canonical_message_id: str | None,
    ) -> bool:
        """Seal a response after canonical persistence, or trace-own it on failure.

        Callers pass the newly persisted assistant ID only after its create
        transaction succeeds. Passing ``None`` deliberately leaves the response
        trace-owned as a sanitized artifact. Settlement remains best-effort and
        never rolls back the already committed conversation message.
        """

        from tldw_chatbook.Chat.console_trace_settlement import (
            ConsoleTraceSettlementCoordinator,
            TraceSettlementRequest,
        )

        if not isinstance(coordinator, ConsoleTraceSettlementCoordinator):
            raise TypeError("coordinator")
        if type(request) is not TraceSettlementRequest:
            raise TypeError("request")
        return coordinator.submit(
            self.db,
            replace(
                cast(TraceSettlementRequest, request),
                canonical_message_id=canonical_message_id,
            ),
        )

    @staticmethod
    def thinking_round_trip_version() -> int:
        """Return the thinking envelope version this local adapter round-trips."""
        return 1

    def persist_console_library_activity(
        self,
        *,
        conversation_id: str,
        contribution: LibraryActivityContribution,
        message_ids: Mapping[str, str],
    ) -> None:
        """Persist one bounded activity batch in a single owned transaction."""
        if not isinstance(contribution, LibraryActivityContribution):
            raise TypeError("contribution must be LibraryActivityContribution")
        with self.db.transaction(immediate=True) as cursor:
            conversation = cursor.execute(
                "SELECT deleted FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None or conversation["deleted"]:
                raise RuntimeError("Durable conversation is unavailable.")
            with _scoped_console_transaction_writer(cursor, conversation_id) as writer:
                contribution.write(
                    writer=writer,
                    conversation_id=conversation_id,
                    message_ids=message_ids,
                )

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

    def get_console_fork_source_message(
        self, message_id: str
    ) -> tuple[int, str] | None:
        """Return one exact persisted source revision/body pair for fork fencing."""

        if type(message_id) is not str or not message_id:
            return None
        message = self.db.get_message_by_id(message_id)
        if message is None or message.get("deleted"):
            return None
        version = message.get("version")
        body = message.get("content")
        if type(version) is not int or version < 1 or type(body) is not str:
            return None
        return version, body

    def get_console_fork_active_leaf(self, conversation_id: str) -> str | None:
        """Return the canonical durable active leaf used by a fork fence."""

        if type(conversation_id) is not str or not conversation_id:
            return None
        active_leaf = self.db.get_conversation_active_leaf(conversation_id)
        return active_leaf if type(active_leaf) is str and active_leaf else None

    def get_console_fork_citation_state(
        self,
        message_id: str,
        revision: int,
        source_body: str,
        target_body: str,
    ) -> tuple[str, str | None]:
        """Return one authoritative citation state for immutable fork staging."""

        repository = self.citation_repository
        if repository is not None and repository.db is self.db:
            return repository.classify_fork_message_owner(
                message_id=message_id,
                message_revision=revision,
                source_message_body=source_body,
                target_message_body=target_body,
            )
        connection = self.db.get_connection()
        message = connection.execute(
            """
            SELECT version, content, deleted
            FROM messages
            WHERE id = ?
            """,
            (message_id,),
        ).fetchone()
        if (
            message is None
            or message["deleted"]
            or message["version"] != revision
            or message["content"] != source_body
        ):
            raise CitationPersistenceUnavailable("fork_source_owner_unverifiable")
        ambiguous = connection.execute(
            """
            SELECT 1
            FROM rag_message_trace_owners
            WHERE message_id = ? AND message_revision = ?
            LIMIT 1
            """,
            (message_id, revision),
        ).fetchone()
        if ambiguous is not None:
            raise CitationPersistenceUnavailable(
                "fork_source_owner_authority_ambiguous"
            )
        return "none", None

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

    def get_conversation_generation_settings(
        self, conversation_id: str
    ) -> ConsoleGenerationSettingsReadResult:
        """Read one conversation's complete safe generation snapshot."""
        if type(conversation_id) is not str or not conversation_id:
            return ConsoleGenerationSettingsReadResult(
                ConsoleGenerationSettingsReadStatus.ABSENT
            )
        record = self.db.get_conversation_by_id(conversation_id)
        if record is None or record.get("deleted"):
            return ConsoleGenerationSettingsReadResult(
                ConsoleGenerationSettingsReadStatus.ABSENT
            )
        return parse_console_generation_settings(record.get("metadata"))

    def update_conversation_generation_settings(
        self,
        *,
        conversation_id: str,
        snapshot: ConsoleGenerationSettingsSnapshot,
        expected_snapshot: ConsoleGenerationSettingsSnapshot | None,
    ) -> ConsoleGenerationSettingsWriteResult:
        """Compare-and-set one complete owned snapshot with one bounded retry.

        A conversation version conflict is retryable only when a fresh read
        proves this codec's complete owned value still equals the caller's
        expected base. The retry then merges against that fresh record so
        unrelated metadata siblings are preserved.
        """
        try:
            merge_console_generation_settings({}, snapshot)
            if expected_snapshot is not None:
                merge_console_generation_settings({}, expected_snapshot)
        except (TypeError, ValueError):
            return ConsoleGenerationSettingsWriteResult(
                ConsoleGenerationSettingsWriteStatus.INVALID
            )

        target = str(conversation_id)
        for attempt in range(2):
            record = self.db.get_conversation_by_id(target)
            if record is None or record.get("deleted"):
                return ConsoleGenerationSettingsWriteResult(
                    ConsoleGenerationSettingsWriteStatus.MISSING
                )
            current = parse_console_generation_settings(record.get("metadata"))
            if current.status is ConsoleGenerationSettingsReadStatus.INVALID:
                return ConsoleGenerationSettingsWriteResult(
                    ConsoleGenerationSettingsWriteStatus.INVALID
                )
            if (
                current.status
                is ConsoleGenerationSettingsReadStatus.UNSUPPORTED_VERSION
            ):
                return ConsoleGenerationSettingsWriteResult(
                    ConsoleGenerationSettingsWriteStatus.UNSUPPORTED_VERSION
                )
            if current.snapshot != expected_snapshot:
                return ConsoleGenerationSettingsWriteResult(
                    ConsoleGenerationSettingsWriteStatus.SUPERSEDED,
                    current.snapshot,
                )
            metadata = merge_console_generation_settings(
                record.get("metadata"),
                snapshot,
            )
            try:
                self.db.update_conversation(
                    target,
                    {
                        "metadata": json.dumps(
                            metadata,
                            allow_nan=False,
                            sort_keys=True,
                        )
                    },
                    expected_version=record["version"],
                )
            except ConflictError:
                if attempt == 0:
                    continue
                fresh_record = self.db.get_conversation_by_id(target)
                if fresh_record is None or fresh_record.get("deleted"):
                    return ConsoleGenerationSettingsWriteResult(
                        ConsoleGenerationSettingsWriteStatus.MISSING
                    )
                fresh = parse_console_generation_settings(fresh_record.get("metadata"))
                if fresh.status is ConsoleGenerationSettingsReadStatus.INVALID:
                    return ConsoleGenerationSettingsWriteResult(
                        ConsoleGenerationSettingsWriteStatus.INVALID
                    )
                if (
                    fresh.status
                    is ConsoleGenerationSettingsReadStatus.UNSUPPORTED_VERSION
                ):
                    return ConsoleGenerationSettingsWriteResult(
                        ConsoleGenerationSettingsWriteStatus.UNSUPPORTED_VERSION
                    )
                if fresh.snapshot != expected_snapshot:
                    return ConsoleGenerationSettingsWriteResult(
                        ConsoleGenerationSettingsWriteStatus.SUPERSEDED,
                        fresh.snapshot,
                    )
                raise
            return ConsoleGenerationSettingsWriteResult(
                ConsoleGenerationSettingsWriteStatus.WRITTEN,
                snapshot,
            )
        raise AssertionError("Unreachable generation-settings retry state.")

    def update_conversation_context_policy(
        self,
        *,
        conversation_id: str,
        overrides: ConsoleContextPolicyOverrides,
        expected_revision: int | None | object = (
            _CONTEXT_POLICY_EXPECTED_REVISION_UNSET
        ),
    ) -> int | None | ContextPolicyWriteResult:
        """Persist context policy, optionally guarding its owned revision.

        Omitting ``expected_revision`` preserves the established unconditional
        caller contract. Settings Apply passes an explicit revision, including
        ``None`` for an absent row, and receives the typed CAS result.
        """
        if expected_revision is not _CONTEXT_POLICY_EXPECTED_REVISION_UNSET:
            return self.context_repository.save_policy_if_revision(
                conversation_id,
                overrides,
                expected_revision=expected_revision,  # type: ignore[arg-type]
            )
        return self.context_repository.save_policy(conversation_id, overrides)

    def update_conversation_thinking_history_policy(
        self,
        *,
        conversation_id: str,
        policy: str,
    ) -> bool:
        """Persist one normalized conversation-owned thinking replay policy."""

        version = self.get_conversation_version(conversation_id)
        if version is None:
            return False
        return bool(
            self.db.update_conversation(
                conversation_id,
                {"thinking_history_policy": policy},
                expected_version=version,
            )
        )

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

    def validate_console_conversation_identity(
        self,
        *,
        runtime_backend: str,
        assistant_kind: str | None,
        assistant_id: str | None,
        assistant_authority_id: str | None,
        persona_memory_mode: str | None,
        character_id: int | None,
    ) -> tuple[str, str | None, str | None, int | None, str | None, str | None]:
        """Require a Console identity to equal this database's canonical form.

        Args:
            runtime_backend: Exact local or server runtime value.
            assistant_kind: Exact generic, character, persona, or null kind.
            assistant_id: Exact stable assistant identifier.
            assistant_authority_id: Exact destination authority identifier.
            persona_memory_mode: Exact persona memory mode.
            character_id: Exact local numeric character identifier.

        Returns:
            The database-normalized identity tuple when it equals the input.

        Raises:
            ValueError: If the database rejects or would normalize the identity.
        """
        normalized = self.db._normalize_conversation_identity(
            runtime_backend=runtime_backend,
            assistant_kind=assistant_kind,
            assistant_id=assistant_id,
            assistant_authority_id=assistant_authority_id,
            persona_memory_mode=persona_memory_mode,
            character_id=character_id,
        )
        candidate = (
            runtime_backend,
            assistant_kind,
            assistant_id,
            character_id,
            persona_memory_mode,
            assistant_authority_id,
        )
        if normalized != candidate:
            raise ValueError("Console conversation identity is not canonical.")
        return normalized

    def create_conversation(
        self,
        *,
        conversation_id: str | None = None,
        root_id: str | None = None,
        parent_conversation_id: str | None = None,
        forked_from_message_id: str | None = None,
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
            root_id: Optional canonical conversation root identity.
            parent_conversation_id: Optional durable source conversation identity.
            forked_from_message_id: Optional persisted source boundary identity.
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
            "root_id": root_id,
            "parent_conversation_id": parent_conversation_id,
            "forked_from_message_id": forked_from_message_id,
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
        context_policy_overrides: ConsoleContextPolicyOverrides | None = None,
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
                if context_policy_overrides is not None:
                    context_result = self.context_repository.save_policy_if_revision(
                        acceptance.conversation_id,
                        context_policy_overrides,
                        expected_revision=None,
                    )
                    if context_result.status is not ContextPolicyWriteStatus.WRITTEN:
                        raise RuntimeError(
                            "Console context settings could not be committed with turn."
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
        project_context_json: str | None = None,
        context_policy_overrides: ConsoleContextPolicyOverrides | None = None,
        contributions: Sequence[ConsoleTransactionContribution] = (),
        trace_boundary: TraceForkBoundary | None = None,
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
            if trace_boundary is not None:
                self._console_trace_repository.attach_fork_owner(
                    cursor,
                    conversation_id=conversation_id,
                    boundary=trace_boundary,
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
            if context_policy_overrides is not None:
                self.context_repository.save_policy(
                    conversation_id,
                    context_policy_overrides,
                )
            if project_context_json is not None:
                self.db.set_conversation_console_project_context(
                    conversation_id,
                    project_context_json,
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

    def fork_console_conversation_bundle(
        self,
        *,
        snapshot: ConsoleChatForkSnapshot,
        conversation_kwargs: Mapping[str, object],
        policy_candidate: ConsoleLibraryPolicyCandidate,
        project_context_json: str,
    ) -> ConsoleForkCommitResult | None:
        """Commit one validated durable fork as a single SQLite bundle."""

        self._validate_console_fork_persistence_input(
            snapshot,
            policy_candidate=policy_candidate,
            project_context_json=project_context_json,
        )
        if not snapshot.durable:
            return None
        target_id = snapshot.fork_conversation_id
        if target_id is None:
            raise ValueError("Durable fork conversation id is unavailable.")
        prepared_conversation = self._fork_conversation_kwargs(
            snapshot,
            conversation_kwargs,
        )
        self.validate_workspace_target(**prepared_conversation)
        with self.db.transaction(immediate=True) as cursor:
            committed = self._resolve_console_fork_commit_cursor(cursor, snapshot)
            if committed is not None:
                return committed
            root_id, parent_id, boundary_id = self._recheck_fork_source(
                cursor,
                snapshot,
            )
            created_id = self.create_conversation(
                conversation_id=target_id,
                root_id=root_id,
                parent_conversation_id=parent_id,
                forked_from_message_id=boundary_id,
                **prepared_conversation,
            )
            if created_id != target_id:
                raise RuntimeError(
                    "Persistence returned an unexpected conversation id."
                )
            if snapshot.trace_boundary is not None:
                self._console_trace_repository.attach_fork_owner(
                    cursor,
                    conversation_id=target_id,
                    boundary=snapshot.trace_boundary,
                )
            for message in snapshot.messages:
                attachments = [
                    {
                        "position": attachment.position,
                        "data": attachment.data,
                        "mime_type": attachment.mime_type,
                        "display_name": attachment.display_name,
                    }
                    for attachment in message.attachments
                ]
                generation_metadata = [
                    {
                        "position": metadata.position,
                        "prompt": metadata.prompt,
                        "negative_prompt": metadata.negative_prompt,
                        "backend": metadata.backend,
                        "model": metadata.model,
                        "seed": metadata.seed,
                        "style": metadata.style,
                        "params_json": metadata.params_json,
                    }
                    for metadata in message.generation_metadata
                ]
                metadata_json = None
                if message.video_tombstone is not None:
                    video = message.video_tombstone
                    metadata_json = VideoGenerationMetadata(
                        name=f"forked-video-{message.native_message_id}",
                        prompt=video.prompt,
                        negative_prompt=video.negative_prompt,
                        backend=video.backend,
                        model=video.model,
                        seed=video.seed,
                        duration_seconds=video.duration_seconds,
                        fps=video.fps,
                        width=video.width,
                        height=video.height,
                        ratio=video.ratio,
                        source_image_message_id=video.source_image_message_id,
                        container=video.container,
                        is_unavailable_tombstone=True,
                    ).to_json()
                else:
                    metadata_json = encode_console_fork_message_metadata(
                        message.status,
                        attachments[0]["display_name"] if attachments else "",
                        message.trace_turn_id,
                    )
                persisted_id = self.create_message(
                    conversation_id=target_id,
                    sender=message.role.value,
                    content=message.content,
                    message_id=message.persisted_message_id,
                    parent_message_id=message.persisted_parent_id,
                    attachments=attachments,
                    generation_metadata=generation_metadata,
                    metadata_json=metadata_json,
                )
                if persisted_id != message.persisted_message_id:
                    raise RuntimeError("Persistence returned an unexpected message id.")
            self._link_console_fork_citations(cursor, snapshot)
            policy_result = self.console_library_policy_repository.insert(
                target_id,
                policy_candidate,
            )
            if policy_result.status is not ConsoleLibraryPolicyWriteStatus.COMMITTED:
                raise RuntimeError(
                    "Console Library policy could not be committed with fork."
                )
            self.context_repository.save_policy(
                target_id,
                snapshot.configuration.context_policy_overrides,
            )
            self.db.set_conversation_console_project_context(
                target_id,
                project_context_json,
            )
            active_leaf = snapshot.messages[-1].persisted_message_id
            self.db.set_conversation_active_leaf(target_id, active_leaf)
        return ConsoleForkCommitResult(
            conversation_id=target_id,
            active_leaf_message_id=active_leaf,
            message_id_map=self._fork_message_id_map(snapshot),
            policy=policy_result.snapshot,
        )

    def resolve_console_fork_commit(
        self,
        snapshot: ConsoleChatForkSnapshot,
    ) -> ConsoleForkCommitResult | None:
        """Resolve an ambiguous durable fork result without writing."""

        if not isinstance(snapshot, ConsoleChatForkSnapshot):
            raise TypeError("snapshot must be ConsoleChatForkSnapshot")
        if not snapshot.durable:
            return None
        with self.db.transaction() as cursor:
            return self._resolve_console_fork_commit_cursor(cursor, snapshot)

    def _resolve_console_fork_commit_cursor(
        self,
        cursor: Any,
        snapshot: ConsoleChatForkSnapshot,
    ) -> ConsoleForkCommitResult | None:
        target_id = snapshot.fork_conversation_id
        if target_id is None:
            raise ValueError("Durable fork conversation id is unavailable.")
        row = cursor.execute(
            """
            SELECT id, root_id, parent_conversation_id, forked_from_message_id,
                   title, active_leaf_message_id, deleted,
                   console_project_context_json
            FROM conversations WHERE id = ?
            """,
            (target_id,),
        ).fetchone()
        if row is None:
            return None
        expected_parent = snapshot.source_conversation_id
        expected_boundary = snapshot.source_boundary_persisted_message_id
        if expected_parent is None:
            expected_root = target_id
        else:
            source = cursor.execute(
                "SELECT root_id FROM conversations WHERE id = ?",
                (expected_parent,),
            ).fetchone()
            expected_root = source["root_id"] if source is not None else None
        active_leaf = snapshot.messages[-1].persisted_message_id
        expected_identity = (
            target_id,
            expected_root,
            expected_parent,
            expected_boundary,
            snapshot.title,
            active_leaf,
            0,
        )
        actual_identity = (
            row["id"],
            row["root_id"],
            row["parent_conversation_id"],
            row["forked_from_message_id"],
            row["title"],
            row["active_leaf_message_id"],
            row["deleted"],
        )
        if actual_identity != expected_identity:
            raise RuntimeError("Console fork target identity collision.")
        for message in snapshot.messages:
            persisted = cursor.execute(
                """
                SELECT conversation_id, parent_message_id, sender, content, deleted
                FROM messages WHERE id = ?
                """,
                (message.persisted_message_id,),
            ).fetchone()
            if persisted is None or tuple(persisted) != (
                target_id,
                message.persisted_parent_id,
                message.role.value,
                message.content,
                0,
            ):
                raise RuntimeError("Console fork target identity collision.")
        if snapshot.trace_boundary is not None and not (
            self._console_trace_repository.fork_owner_matches_boundary(
                cursor,
                conversation_id=target_id,
                boundary=snapshot.trace_boundary,
            )
        ):
            raise RuntimeError("Console fork target identity collision.")
        policy = self.console_library_policy_repository.read(target_id).durable_policy
        if policy is None:
            raise RuntimeError("Console fork target identity collision.")
        return ConsoleForkCommitResult(
            conversation_id=target_id,
            active_leaf_message_id=active_leaf,
            message_id_map=self._fork_message_id_map(snapshot),
            policy=policy,
            already_committed=True,
        )

    @staticmethod
    def _fork_message_id_map(snapshot: ConsoleChatForkSnapshot) -> dict[str, str]:
        return {
            message.native_message_id: message.persisted_message_id
            for message in snapshot.messages
            if message.persisted_message_id is not None
        }

    def _recheck_fork_source(
        self,
        cursor: Any,
        snapshot: ConsoleChatForkSnapshot,
    ) -> tuple[str, str | None, str | None]:
        source_id = snapshot.source_conversation_id
        if source_id is None:
            return snapshot.fork_conversation_id or "", None, None
        source = cursor.execute(
            """
            SELECT root_id, version, deleted, active_leaf_message_id
            FROM conversations WHERE id = ?
            """,
            (source_id,),
        ).fetchone()
        if (
            source is None
            or source["deleted"]
            or type(snapshot.source_conversation_version) is not int
            or source["version"] != snapshot.source_conversation_version
            or source["active_leaf_message_id"]
            != snapshot.source_active_leaf_persisted_message_id
        ):
            raise RuntimeError("Console fork source changed.")
        previous_source_id = None
        for message in snapshot.messages:
            row = cursor.execute(
                """
                SELECT conversation_id, parent_message_id, version, content, deleted
                FROM messages WHERE id = ?
                """,
                (message.source_persisted_message_id,),
            ).fetchone()
            if (
                row is None
                or row["conversation_id"] != source_id
                or row["parent_message_id"] != previous_source_id
                or row["version"] != message.source_persisted_revision
                or row["deleted"]
                or row["content"] != message.source_persisted_content
            ):
                raise RuntimeError("Console fork source changed.")
            previous_source_id = message.source_persisted_message_id
        if previous_source_id != snapshot.source_boundary_persisted_message_id:
            raise RuntimeError("Console fork source changed.")
        active_lineage_id = snapshot.source_active_leaf_persisted_message_id
        seen: set[str] = set()
        for _ in range(CONSOLE_FORK_SOURCE_LINEAGE_MAX_DEPTH):
            if active_lineage_id in seen or not active_lineage_id:
                raise RuntimeError("Console fork source changed.")
            seen.add(active_lineage_id)
            active_row = cursor.execute(
                """
                SELECT conversation_id, parent_message_id, deleted
                FROM messages WHERE id = ?
                """,
                (active_lineage_id,),
            ).fetchone()
            if (
                active_row is None
                or active_row["conversation_id"] != source_id
                or active_row["deleted"]
            ):
                raise RuntimeError("Console fork source changed.")
            if active_lineage_id == previous_source_id:
                break
            active_lineage_id = active_row["parent_message_id"]
        else:
            raise RuntimeError("Console fork source changed.")
        return source["root_id"], source_id, previous_source_id

    def _link_console_fork_citations(
        self,
        cursor: Any,
        snapshot: ConsoleChatForkSnapshot,
    ) -> None:
        target_by_source = {
            message.source_persisted_message_id: message
            for message in snapshot.messages
            if message.source_persisted_message_id is not None
        }
        for link in snapshot.citation_links:
            target = target_by_source.get(link.source_persisted_message_id)
            if target is None:
                raise CitationPersistenceUnavailable("fork_citation_owner_missing")
            if link.state != "active_required":
                continue
            repository = self.citation_repository
            if repository is None or repository.db is not self.db:
                raise CitationPersistenceUnavailable("citation_repository_unavailable")
            repository.link_fork_message_owner(
                cursor,
                source_message_id=link.source_persisted_message_id,
                source_message_revision=link.source_revision,
                source_message_body=target.source_persisted_content or "",
                target_message_id=target.persisted_message_id or "",
                target_message_revision=1,
                target_message_body=target.content,
                confirmed_state=link.state,
                confirmed_trace_id=link.trace_id,
            )

    @staticmethod
    def _fork_conversation_kwargs(
        snapshot: ConsoleChatForkSnapshot,
        conversation_kwargs: Mapping[str, object],
    ) -> dict[str, object]:
        configuration = snapshot.configuration
        global_scope = configuration.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID
        prepared: dict[str, object] = {
            "conversation_title": snapshot.title,
            "scope_type": "global" if global_scope else "workspace",
            "workspace_id": None if global_scope else configuration.workspace_id,
            "system_prompt": configuration.settings.system_prompt,
            "runtime_backend": configuration.runtime_backend,
            "assistant_kind": configuration.assistant_kind,
            "assistant_id": configuration.assistant_id,
            "assistant_authority_id": configuration.assistant_authority_id,
            "persona_memory_mode": configuration.persona_memory_mode,
            "character_id": configuration.character_id,
            "character_name": configuration.character_name,
            "speech_preferences": configuration.speech_preferences,
            "thinking_history_policy": configuration.thinking_history_policy,
        }
        if dict(conversation_kwargs) != prepared:
            raise ValueError("Console fork configuration changed.")
        metadata: dict[str, object] = {}
        serialized_settings = asdict(configuration.settings)
        if configuration.ephemeral_endpoint_policy is not None:
            serialized_settings.pop("base_url", None)
        metadata["console_session_settings"] = {
            "version": 1,
            **serialized_settings,
            "pinned_prefill": None,
        }
        metadata = merge_console_generation_settings(
            metadata,
            snapshot_from_session_settings(configuration.settings),
        )
        if configuration.rag_scope is not None:
            metadata["rag_scope"] = serialize_scope(configuration.rag_scope)
        if (
            configuration.user_display_name_override is not None
            or configuration.character_system_template is not None
        ):
            metadata = json.loads(
                merge_console_roleplay_context(
                    metadata,
                    ConsoleRoleplayContext(
                        user_name_override=(configuration.user_display_name_override),
                        character_system_template=(
                            configuration.character_system_template
                        ),
                    ),
                )
            )
        prepared["metadata"] = metadata or None
        return prepared

    @staticmethod
    def _validate_console_fork_persistence_input(
        snapshot: ConsoleChatForkSnapshot,
        *,
        policy_candidate: ConsoleLibraryPolicyCandidate,
        project_context_json: str,
    ) -> None:
        if not isinstance(snapshot, ConsoleChatForkSnapshot):
            raise TypeError("snapshot must be ConsoleChatForkSnapshot")
        if policy_candidate != snapshot.configuration.library_policy:
            raise ValueError("Console fork Library policy changed.")
        if project_context_json != encode_project_context_json(
            snapshot.configuration.project_instruction_state
        ):
            raise ValueError("Console fork project context changed.")
        if not snapshot.messages:
            raise ValueError("Console fork must contain a message.")
        durable_ids = [message.persisted_message_id for message in snapshot.messages]
        if snapshot.durable:
            if (
                not snapshot.fork_conversation_id
                or any(not message_id for message_id in durable_ids)
                or len(set(durable_ids)) != len(durable_ids)
            ):
                raise ValueError("Console fork durable identities are invalid.")
        elif snapshot.fork_conversation_id is not None or any(durable_ids):
            raise ValueError("Temporary fork cannot carry durable identities.")
        if not snapshot.durable and (
            snapshot.source_conversation_id is not None
            or snapshot.source_conversation_version is not None
            or snapshot.source_active_leaf_persisted_message_id is not None
            or snapshot.source_boundary_persisted_message_id is not None
            or snapshot.citation_links
            or any(
                message.source_persisted_message_id is not None
                or message.source_persisted_revision is not None
                or message.source_persisted_content is not None
                for message in snapshot.messages
            )
        ):
            raise ValueError("Temporary fork citation identity is invalid.")
        source_carriers = tuple(
            (
                message.source_persisted_message_id,
                message.source_persisted_revision,
                message.source_persisted_content,
            )
            for message in snapshot.messages
        )
        if snapshot.source_conversation_id is not None:
            if (
                type(snapshot.source_conversation_version) is not int
                or not snapshot.source_active_leaf_persisted_message_id
                or not snapshot.source_boundary_persisted_message_id
                or any(
                    type(source_id) is not str
                    or not source_id
                    or type(revision) is not int
                    or type(content) is not str
                    for source_id, revision, content in source_carriers
                )
            ):
                raise ValueError("Console fork durable source fence is invalid.")
        elif snapshot.durable and (
            snapshot.source_conversation_version is not None
            or snapshot.source_active_leaf_persisted_message_id is not None
            or snapshot.source_boundary_persisted_message_id is not None
            or any(carrier != (None, None, None) for carrier in source_carriers)
            or snapshot.citation_links
        ):
            raise ValueError(
                "Unsaved fork source cannot carry durable source identity."
            )
        expected_citations = {
            (
                message.source_persisted_message_id,
                message.source_persisted_revision,
            )
            for message in snapshot.messages
            if message.source_persisted_message_id is not None
        }
        actual_citations = {
            (link.source_persisted_message_id, link.source_revision)
            for link in snapshot.citation_links
        }
        if (
            len(actual_citations) != len(snapshot.citation_links)
            or actual_citations != expected_citations
            or any(
                link.state not in {"active_required", "unavailable", "none"}
                for link in snapshot.citation_links
            )
            or any(
                (
                    link.state == "active_required"
                    and (type(link.trace_id) is not str or not link.trace_id)
                )
                or (link.state != "active_required" and link.trace_id is not None)
                for link in snapshot.citation_links
            )
        ):
            raise ValueError("Console fork citation states are invalid.")
        prior_native = None
        prior_persisted = None
        image_ids: set[str] = set()
        for message in snapshot.messages:
            if message.status not in {"complete", "stopped", "failed"}:
                raise ValueError("Console fork message status is invalid.")
            if (
                message.native_parent_id != prior_native
                or message.persisted_parent_id != prior_persisted
            ):
                raise ValueError("Console fork lineage is invalid.")
            if message.video_tombstone is not None:
                if (
                    message.attachments
                    or message.generation_metadata
                    or message.status != "complete"
                    or message.content != CONSOLE_FORK_VIDEO_TOMBSTONE_CONTENT
                    or message.video_tombstone.owner_native_message_id
                    != message.native_message_id
                    or message.video_tombstone.owner_persisted_message_id
                    != message.persisted_message_id
                    or (
                        message.video_tombstone.source_image_message_id is not None
                        and message.video_tombstone.source_image_message_id
                        not in image_ids
                    )
                ):
                    raise ValueError("Console fork video tombstone is invalid.")
            if message.generation_metadata and len(message.generation_metadata) != len(
                message.attachments
            ):
                raise ValueError("Console fork generation metadata is invalid.")
            has_image = False
            for position, attachment in enumerate(message.attachments):
                if (
                    attachment.owner_native_message_id != message.native_message_id
                    or attachment.owner_persisted_message_id
                    != message.persisted_message_id
                    or attachment.position != position
                    or type(attachment.data) is not bytes
                    or not attachment.data
                    or len(attachment.data) > MAX_ATTACHMENT_BYTES
                    or type(attachment.mime_type) is not str
                    or not attachment.mime_type
                    or type(attachment.display_name) is not str
                ):
                    raise ValueError("Console fork attachment is invalid.")
                if attachment.mime_type.startswith("image/"):
                    validate_console_fork_image_payload(
                        attachment.data,
                        attachment.mime_type,
                    )
                    has_image = True
            for position, metadata in enumerate(message.generation_metadata):
                if (
                    metadata.owner_native_message_id != message.native_message_id
                    or metadata.owner_persisted_message_id
                    != message.persisted_message_id
                    or metadata.position != position
                    or not has_image
                ):
                    raise ValueError("Console fork generation metadata is invalid.")
                try:
                    fingerprint_console_fork_selected_image(
                        message.attachments[position],
                        metadata,
                    )
                except (TypeError, ValueError):
                    raise ValueError(
                        "Console fork generation metadata is invalid."
                    ) from None
            if has_image:
                image_id = (
                    message.persisted_message_id
                    if snapshot.durable
                    else message.native_message_id
                )
                if image_id is None:
                    raise ValueError("Console fork image identity is invalid.")
                image_ids.add(image_id)
            prior_native = message.native_message_id
            prior_persisted = message.persisted_message_id

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

    def update_conversation_console_session_settings(
        self,
        *,
        conversation_id: str,
        settings: ConsoleSessionSettings,
    ) -> bool:
        """Merge the current Console settings snapshot into conversation metadata.

        The top-level conversation system prompt and pinned-prefill metadata remain
        canonical on resume. This snapshot preserves the rest of the settings that
        changed after the conversation's first durable write.

        Args:
            conversation_id: Durable conversation identifier.
            settings: Complete current Console session settings.

        Returns:
            True when persisted; False when the conversation no longer exists.
        """

        for attempt in range(2):
            record = self.db.get_conversation_by_id(str(conversation_id))
            if record is None:
                return False
            metadata = _initial_metadata_object(record.get("metadata") or {})
            metadata["console_session_settings"] = {
                "version": 1,
                **asdict(settings),
            }
            try:
                self.db.update_conversation(
                    str(conversation_id),
                    {"metadata": json.dumps(metadata)},
                    expected_version=record["version"],
                )
                return True
            except ConflictError:
                if attempt == 1:
                    raise
        return False

    def adopt_console_session_endpoint_settings(
        self,
        *,
        conversation_id: str,
        settings: ConsoleSessionSettings,
    ) -> ConsoleEndpointAdoptionReceipt:
        """Persist safe generation values and return an exact rollback receipt.

        The caller owns the verified endpoint separately in process memory. The
        endpoint-free generation codec is the only metadata owner changed here;
        the complete settings codec (which includes ``base_url``) is deliberately
        reserved for ordinary Console settings persistence.
        """

        for attempt in range(2):
            record = self.db.get_conversation_by_id(str(conversation_id))
            if record is None:
                raise RuntimeError("Console conversation disappeared during adoption")
            before_metadata = record.get("metadata")
            metadata = _initial_metadata_object(before_metadata or {})
            metadata = merge_console_generation_settings(
                metadata,
                snapshot_from_session_settings(settings),
            )
            written_metadata = json.dumps(metadata)
            version = record.get("version")
            if type(version) is not int:
                raise RuntimeError("Console conversation version is invalid")
            try:
                self.db.update_conversation(
                    str(conversation_id),
                    {"metadata": written_metadata},
                    expected_version=version,
                )
            except ConflictError:
                if attempt == 1:
                    raise
                continue
            return ConsoleEndpointAdoptionReceipt(
                conversation_id=str(conversation_id),
                before_metadata=before_metadata,
                written_metadata=written_metadata,
                written_version=version + 1,
            )
        raise RuntimeError("Console endpoint adoption could not be persisted")

    def rollback_console_session_endpoint_adoption(
        self,
        *,
        receipt: ConsoleEndpointAdoptionReceipt,
    ) -> bool:
        """Restore exact metadata only while the adoption still owns the row."""

        if not isinstance(receipt, ConsoleEndpointAdoptionReceipt):
            raise TypeError("Console endpoint adoption receipt is required")
        record = self.db.get_conversation_by_id(receipt.conversation_id)
        if record is None:
            return False
        if (
            record.get("version") != receipt.written_version
            or record.get("metadata") != receipt.written_metadata
        ):
            return False
        try:
            self.db.update_conversation(
                receipt.conversation_id,
                {"metadata": receipt.before_metadata},
                expected_version=receipt.written_version,
            )
        except ConflictError:
            return False
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
        expected_version: int | None = None,
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
            expected_version is not None
            and current_message.get("version") != expected_version
        ):
            return False
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
            update_data["provider_continuation_json"] = None

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

        def coordinated_update() -> bool:
            update_version = (
                current_message["version"]
                if expected_version is None
                else expected_version
            )
            if attachments is None:
                return bool(
                    self.db.update_message(
                        message_id,
                        update_data,
                        expected_version=update_version,
                        preserve_provider_continuation=preserve_provider_continuation,
                        preserve_descendants=preserve_descendants,
                    )
                )
            return bool(
                self.db.update_message_with_attachments(
                    message_id,
                    update_data,
                    expected_version=update_version,
                    attachments=extra_rows,
                    preserve_provider_continuation=preserve_provider_continuation,
                    preserve_descendants=preserve_descendants,
                )
            )

        if citation_repository is not None:
            # IMMEDIATE (task-21100): `transaction(immediate=...)` is honored
            # only at depth 0, so this OUTER wrapper decides the begin mode for
            # the nested hot messages writers -- left DEFERRED it re-opens the
            # snapshot-upgrade "database is locked" window their own IMMEDIATE
            # closes (see add_message's scoping comment).
            with self.db.transaction(immediate=True) as cursor:
                result = coordinated_update()
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
            with self.db.transaction(immediate=True) as cursor:
                result = coordinated_update()
            return result

        return coordinated_update()

    def read_canonical_generation_projection(
        self, message_id: str
    ) -> Mapping[str, Any] | None:
        """Read the canonical fields required by body-only generation writers.

        This explicit capability lets adapters hide their database handle while
        still giving the Console a versioned, deletion-aware CAS boundary.
        """
        row = self.db.get_message_by_id(message_id)
        if row is None:
            return None
        return {
            "id": row.get("id"),
            "conversation_id": row.get("conversation_id"),
            "sender": row.get("sender"),
            "version": row.get("version"),
            "deleted": row.get("deleted"),
            "content": row.get("content"),
            "image_data": row.get("image_data"),
            "image_mime_type": row.get("image_mime_type"),
            "assistant_generation_state": row.get("assistant_generation_state"),
            "thinking_blocks_json": row.get("thinking_blocks_json"),
            "provider_continuation_json": row.get("provider_continuation_json"),
            "usage_json": row.get("usage_json"),
            "metadata_json": row.get("metadata_json"),
        }

    def read_canonical_generation_projection_bundle(
        self, message_id: str
    ) -> Mapping[str, Any] | None:
        """Read a generation row and ordered sidecars from one SQLite snapshot."""

        with self.db.transaction(immediate=True):
            row = self.read_canonical_generation_projection(message_id)
            if row is None:
                return None
            attachments = self.db.get_attachments_for_messages([message_id]).get(
                message_id, []
            )
            generation_metadata = self.db.get_generation_metadata_for_messages(
                [message_id]
            ).get(message_id, [])
            return {
                "message": dict(row),
                "attachments": [dict(item) for item in attachments],
                "generation_metadata": [dict(item) for item in generation_metadata],
            }

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
            feedback: Optional feedback value persisted with the initial row.
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
        citation_repository = self.citation_repository
        if citation_write is not None:
            if citation_repository is None:
                raise CitationPersistenceUnavailable("citation_repository_unavailable")
            if citation_repository.db is not self.db:
                raise CitationPersistenceUnavailable(
                    "citation_repository_database_mismatch"
                )
            prepared_citation = citation_repository.prepare_write(citation_write)

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
                    created_message_id = self.db.add_message_with_semantic_sidecars(
                        message_payload,
                        attachments=extra_rows if attachments is not None else (),
                        generation_metadata=(
                            [dict(row) for row in generation_metadata]
                            if generation_metadata is not None
                            else ()
                        ),
                        feedback=feedback,
                    )
                    if created_message_id is None:
                        raise RuntimeError("Message persistence did not return an ID.")
                # TASK-22226: only ``version`` (message_revision) is consumed
                # here; write_prepared independently re-validates it against
                # the messages row inside this same transaction.
                created_message = self.db.get_message_by_id_without_blob(
                    created_message_id
                )
                if created_message is None or citation_repository is None:
                    raise RuntimeError("Committed message could not be reloaded.")
                citation_repository.write_prepared(
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
                created_message_id = self.db.add_message_with_semantic_sidecars(
                    message_payload,
                    attachments=extra_rows if attachments is not None else (),
                    generation_metadata=(
                        [dict(row) for row in generation_metadata]
                        if generation_metadata is not None
                        else ()
                    ),
                    feedback=feedback,
                )
        else:
            created_message_id = self.db.add_message_with_semantic_sidecars(
                message_payload,
                feedback=feedback,
            )
        if created_message_id is None:
            raise RuntimeError("Message persistence did not return an ID.")
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
