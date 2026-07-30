"""Native Console chat session store and persistence facade."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.citation_trace_models import (
    ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX,
    SealedCitationWrite,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
)
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    DEFAULT_CONSOLE_SESSION_TITLE,
    ConsoleChatMessage,
    ConsoleCitationPresentation,
    ConsoleMessageFeedback,
    ConsoleMessageRole,
    ConsoleMessageStatus,
    ConsoleVariant,
    ConsoleVariantSet,
    ConsoleWorkspaceContext,
    GenerationVariantMeta,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_speech import (
    ConsoleSpeechSnapshotRejected,
    ConsoleSpeechSnapshotRejectionCode,
    TTSMessageSpeechSnapshot,
)
from tldw_chatbook.Chat.rag_scope import RagScope, SessionScopeHolder
from tldw_chatbook.TTS.profile_errors import ProfileValidationError
from tldw_chatbook.TTS.profile_types import CharacterRef

#: Maximum number of attachments a Console session may stage before send.
MAX_PENDING_ATTACHMENTS = 5

TerminalCitationFinalizer = Callable[[str], SealedCitationWrite | None]


@dataclass(frozen=True)
class _VariantStreamBase:
    """Pre-regenerate snapshot captured by ``begin_variant_stream``.

    Carries both the visible content *and* the message's status at the
    moment regeneration began, so a failed regenerate can restore the
    message to exactly the state it was in before -- not just its content.
    """

    content: str
    prior_status: ConsoleMessageStatus


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
    ) -> str:
        """Create a persisted message and return its ID.

        ``attachments``, when given, covers ALL positions (0..N-1) and is
        authoritative over the scalar ``image_data``/``image_mime_type``
        kwargs; ``None`` leaves the pre-split legacy behavior unchanged.
        Optional: fakes used in tests may omit this parameter entirely.

        ``citation_write``, when present, is committed atomically with the
        message by citation-aware adapters. Narrow test fakes may omit this
        optional parameter entirely.
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
    ) -> bool:
        """Update persisted message content.

        ``attachments`` follows the same split-addressing contract as
        ``create_message``; ``None`` (the Console store's edit path always
        passes this) leaves attachments untouched. Optional: fakes used in
        tests may omit this parameter entirely.
        """

    def get_message_version(self, message_id: str) -> int | None:
        """Return the current positive durable row version, if trustworthy."""

    def update_conversation_system_prompt(
        self,
        *,
        conversation_id: str,
        system_prompt: str | None,
    ) -> bool:
        """Persist a changed system prompt for an already-saved conversation."""

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
    draft: str = ""
    updated_at: str = field(default_factory=_utc_now_iso)
    pending_attachments: list[PendingAttachment] = field(default_factory=list)
    one_shot_prefill: str | None = None
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


class ConsoleChatStore:
    """Manage native Console sessions and messages before UI integration."""

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
        self.active_session_id: str | None = None
        self._sessions: dict[str, ConsoleChatSession] = {}
        #: Derived VIEW = the current active path only (root -> active leaf).
        #: Written ONLY by ``_recompute_active_path`` (single-writer invariant);
        #: every other reader/writer of the tree goes through the maps below.
        self._messages_by_session: dict[str, list[ConsoleChatMessage]] = {}
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
        self._pending_persistence_message_ids: set[str] = set()
        self._terminal_citation_finalizers: dict[str, TerminalCitationFinalizer] = {}
        self._provisional_terminal_selection_ids: set[str] = set()
        self._terminal_persistence_deferred_ids: set[str] = set()
        self._stream_chunks_by_message: dict[str, list[str]] = {}
        self._stream_materialized_counts: dict[str, int] = {}
        self._sync_v2_message_versions: dict[str, str] = {}
        self._variant_stream_bases: dict[str, _VariantStreamBase] = {}
        # Ephemeral fence for issued speech snapshots. It deliberately lives
        # outside ConsoleChatMessage so it is neither persisted nor restored.
        self._message_speech_revisions: dict[str, int] = {}

    def ensure_session(
        self,
        *,
        title: str = DEFAULT_CONSOLE_SESSION_TITLE,
        workspace_id: str | None = None,
        settings: ConsoleSessionSettings | None = None,
    ) -> ConsoleChatSession:
        """Return the active session, creating one when needed."""
        if self.active_session_id is not None:
            return self._sessions[self.active_session_id]
        return self.create_session(
            title=title, workspace_id=workspace_id, settings=settings
        )

    def create_session(
        self,
        *,
        title: str = DEFAULT_CONSOLE_SESSION_TITLE,
        workspace_id: str | None = None,
        settings: ConsoleSessionSettings | None = None,
        runtime_backend: str = "local",
        assistant_kind: str | None = "generic",
        assistant_id: str | None = "console",
        assistant_authority_id: str | None = None,
        character_id: int | None = None,
        character_name: str | None = None,
    ) -> ConsoleChatSession:
        """Create and activate a new native Console session."""
        session = ConsoleChatSession(
            title=title,
            workspace_id=workspace_id or self.workspace_context.active_workspace_id,
            settings=settings,
            runtime_backend=runtime_backend,
            assistant_kind=assistant_kind,
            assistant_id=assistant_id,
            assistant_authority_id=assistant_authority_id,
            character_id=character_id,
            character_name=character_name,
        )
        self._sessions[session.id] = session
        self._messages_by_session[session.id] = []
        self._nodes_by_session[session.id] = {}
        self._children_by_parent[session.id] = {}
        self._active_leaf_by_session[session.id] = None
        self._context_summary_by_session[session.id] = (None, None)
        self.active_session_id = session.id
        return session

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
        )
        session.persisted_conversation_id = str(persisted_conversation_id)
        self._ingest_full_tree(
            session.id,
            all_nodes,
            active_leaf_persisted_id=active_leaf_persisted_id,
        )
        self._hydrate_generation_metadata_from_persistence(session.id)
        return session

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
        for message in messages:
            if message.role is ConsoleMessageRole.TOOL:
                self._message_session_index.setdefault(message.id, session_id)
                overlay.append(message)
            else:
                overlay.append(nodes.get(message.id, message))
        self._messages_by_session[session_id] = overlay

    def switch_session(self, session_id: str) -> ConsoleChatSession:
        """Activate an existing session."""
        session = self._session_or_raise(session_id)
        self.active_session_id = session.id
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
        session_ids = list(self._sessions.keys())
        closed_index = session_ids.index(session_id)

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
            self._message_speech_revisions.pop(message_id, None)
            self._native_parent_by_message.pop(message_id, None)

        self._messages_by_session.pop(session_id, None)
        self._nodes_by_session.pop(session_id, None)
        self._children_by_parent.pop(session_id, None)
        self._active_leaf_by_session.pop(session_id, None)
        self._context_summary_by_session.pop(session_id, None)
        self._sessions.pop(session_id, None)

        if self.active_session_id != session_id:
            return self._sessions.get(self.active_session_id or "")

        remaining_sessions = list(self._sessions.values())
        if not remaining_sessions:
            self.active_session_id = None
            return None

        next_index = min(closed_index, len(remaining_sessions) - 1)
        next_session = remaining_sessions[next_index]
        self.active_session_id = next_session.id
        return next_session

    def sessions(self) -> list[ConsoleChatSession]:
        """Return native Console sessions in creation order."""
        return list(self._sessions.values())

    def session_settings(self, session_id: str) -> ConsoleSessionSettings | None:
        """Return in-memory settings for a native Console session."""
        return self._session_or_raise(session_id).settings

    def session_workspace_id(self, session_id: str) -> str:
        """Return the workspace id a native Console session is bound to.

        Used by ``ConsoleAgentBridge.run_reply`` (task-6, settings-
        workspaces-folder-roots spec §3) to thread the RUNNING session's
        own workspace into ``BuiltinToolProvider`` -- never whatever
        workspace happens to be active in the UI by the time a tool
        actually fires.
        """
        return self._session_or_raise(session_id).workspace_id

    def replace_session_settings(
        self,
        session_id: str,
        settings: ConsoleSessionSettings,
    ) -> ConsoleChatSession:
        """Replace in-memory settings for a native Console session."""
        session = self._session_or_raise(session_id)
        session.settings = settings
        return session

    def session_draft(self, session_id: str) -> str:
        """Return the in-memory composer draft for a native Console session."""
        return self._session_or_raise(session_id).draft

    def set_session_draft(self, session_id: str, draft: str) -> ConsoleChatSession:
        """Replace the in-memory composer draft for a native Console session."""
        session = self._session_or_raise(session_id)
        session.draft = draft
        return session

    def session_one_shot_prefill(self, session_id: str) -> str | None:
        """Return the armed one-shot response prefill for a session, if any."""
        return self._session_or_raise(session_id).one_shot_prefill

    def set_session_one_shot_prefill(
        self, session_id: str, prefill: str | None
    ) -> ConsoleChatSession:
        """Arm (or clear, with ``None``) the one-shot response prefill."""
        session = self._session_or_raise(session_id)
        session.one_shot_prefill = prefill
        return session

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
        self.active_session_id = None
        self._sessions.clear()
        self._messages_by_session.clear()
        self._message_session_index.clear()
        self._pending_persistence_message_ids.clear()
        self._terminal_citation_finalizers.clear()
        self._provisional_terminal_selection_ids.clear()
        self._terminal_persistence_deferred_ids.clear()
        self._stream_chunks_by_message.clear()
        self._stream_materialized_counts.clear()
        self._sync_v2_message_versions.clear()
        # Pre-existing bug fixed while here: the regenerate base snapshots were
        # never cleared on restore, leaking across a state replacement.
        self._variant_stream_bases.clear()
        self._message_speech_revisions.clear()
        self._nodes_by_session.clear()
        self._children_by_parent.clear()
        self._native_parent_by_message.clear()
        self._active_leaf_by_session.clear()
        self._context_summary_by_session.clear()

        messages_by_session = messages_by_session or {}
        for session in restored_sessions:
            self._sessions[session.id] = replace(session)
            self._nodes_by_session[session.id] = {}
            self._children_by_parent[session.id] = {}
            self._active_leaf_by_session[session.id] = None
            self._context_summary_by_session[session.id] = (None, None)
            self._messages_by_session[session.id] = []
            self._ingest_linear_messages(
                session.id, messages_by_session.get(session.id, ())
            )

        if active_session_id in self._sessions:
            self.active_session_id = active_session_id
        elif self._sessions:
            self.active_session_id = next(iter(self._sessions))

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
    ) -> ConsoleChatMessage:
        """Append a message; scalar image kwargs become a one-item tuple."""
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
            role=role,
            content=content,
            status=self._initial_status(role=role, content=content),
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
            self._message_session_index[message.id] = session_id
            self._messages_by_session[session_id].append(message)
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

    def get_message(self, message_id: str) -> ConsoleChatMessage:
        """Return a message by native message ID."""
        message = self._message_or_raise(message_id)
        self._materialize_stream_buffer(message)
        return self._snapshot(message)

    def issue_tts_message_speech_snapshot(
        self,
        message_id: str,
    ) -> TTSMessageSpeechSnapshot:
        """Issue a trusted snapshot for one speakable active-path message.

        Args:
            message_id: Native Console message selected by the user.

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
    ) -> str:
        """Revalidate an issued Console speech snapshot against live state.

        Args:
            snapshot: Immutable snapshot previously issued by this store.

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

        message.content = selected_body
        buffer = self._stream_chunks_by_message.get(message.id)
        if buffer is None:
            self._stream_chunks_by_message[message.id] = [selected_body]
        else:
            buffer[:] = [selected_body]
        self._stream_materialized_counts[message.id] = 1
        self._bump_message_speech_revision(message.id)
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
        if message.variants is None:
            message.content = content
        else:
            selected_index = message.variants.selected_index
            message.variants.variants[selected_index] = replace(
                message.variants.variants[selected_index],
                content=content,
            )
            message.content = message.variants.current.content
        self._bump_message_speech_revision(message.id)
        self._persist_existing_message(message)
        return self._snapshot(message)

    def delete_message(self, message_id: str) -> ConsoleChatMessage:
        """Remove a complete Console message from the local transcript."""
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
            self._message_speech_revisions.pop(node_id, None)
        # Only when the deleted branch was on the active path does the leaf move
        # (up to the deleted node's parent); an off-path delete leaves it alone.
        if on_active_path:
            self._active_leaf_by_session[session_id] = parent_native_id
        self._recompute_active_path(session_id)
        return self._snapshot(message)

    def session_id_for_message(self, message_id: str) -> str:
        """Return the owning session ID for a message."""
        if message_id not in self._message_session_index:
            raise KeyError(f"Unknown Console message: {message_id}")
        return self._message_session_index[message_id]

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
        self._active_leaf_by_session[session_id] = message_id
        self._recompute_active_path(session_id)
        self._persist_active_leaf(session_id, message_id)

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
        self._context_summary_by_session[session_id] = (summary, boundary_native_id)
        self._persist_context_summary(session_id, summary, boundary_native_id)

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
        buffer = self._stream_chunks_by_message.setdefault(
            message.id,
            [message.content] if message.content else [],
        )
        buffer.append(chunk)
        message.status = "streaming"
        self._bump_message_speech_revision(message.id)
        return self._snapshot(message)

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

    def mark_message_complete(self, message_id: str) -> ConsoleChatMessage:
        """Mark a message complete and flush final visible content to persistence."""
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
        self._materialize_stream_buffer(message)
        finalizer = self._terminal_citation_finalizers.pop(message.id, None)
        terminal_persistence = (
            finalizer is not None
            or message.id in self._terminal_persistence_deferred_ids
        )
        self.clear_terminal_citation_state(message.id)
        if not terminal_persistence:
            message.status = "complete"
            self._bump_message_speech_revision(message.id)
            self._persist_existing_message(message)
            return self._snapshot(message)

        try:
            if not message.content:
                message.status = "complete"
                self._bump_message_speech_revision(message.id)
                self._persist_existing_message(message)
                return self._snapshot(message)

            citation_write = None
            if finalizer is not None:
                try:
                    citation_write = finalizer(message.content)
                except Exception:
                    logger.warning("terminal_finalizer_unavailable")
            message.status = "complete"
            self._bump_message_speech_revision(message.id)
            session_id = self._message_session_index[message.id]
            try:
                self._persist_new_message(
                    session_id=session_id,
                    message=message,
                    citation_write=citation_write,
                    force_stable_message_id=True,
                    terminal_persistence=True,
                )
            except Exception:
                self._pending_persistence_message_ids.discard(message.id)
                logger.warning("terminal_citation_persistence_abandoned")
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
        self._materialize_stream_buffer(message)
        self.clear_terminal_citation_state(message.id)
        base = self._variant_stream_bases.pop(message.id, None)
        if base is not None:
            message.content = base.content
            message.status = base.prior_status
        else:
            message.status = "stopped"
        self._bump_message_speech_revision(message.id)
        self._persist_existing_message(message)
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
        self._materialize_stream_buffer(message)
        self.clear_terminal_citation_state(message.id)
        base = self._variant_stream_bases.pop(message.id, None)
        if base is not None:
            message.content = base.content
            message.status = base.prior_status
        else:
            message.status = "failed"
        self._bump_message_speech_revision(message.id)
        self._persist_existing_message(message)
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
        message.content = ""
        message.status = "pending"
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        self._bump_message_speech_revision(message.id)
        return self._snapshot(message)

    def add_variant(self, message_id: str, content: str) -> ConsoleChatMessage:
        """Add and select a regenerated variant for an assistant message."""
        message = self._message_or_raise(message_id)
        self._materialize_stream_buffer(message)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can receive variants.")
        if message.variants is None:
            message.variants = ConsoleVariantSet.from_contents(
                turn_id=message.turn_id or message.id,
                contents=[message.content, content],
                selected_index=1,
            )
        else:
            message.variants.variants.append(ConsoleVariant(content=content))
            message.variants.selected_index = len(message.variants.variants) - 1
        message.content = message.variants.current.content
        self._bump_message_speech_revision(message.id)
        self._persist_existing_message(message)
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
        self._materialize_stream_buffer(message)
        self._variant_stream_bases[message.id] = _VariantStreamBase(
            content=message.content,
            prior_status=message.status,
        )
        message.content = ""
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        message.status = "streaming"
        self._bump_message_speech_revision(message.id)
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
        self._materialize_stream_buffer(message)
        new_content = message.content
        base_entry = self._variant_stream_bases.pop(message.id, None)
        base = base_entry.content if base_entry is not None else ""
        if message.variants is None:
            message.variants = ConsoleVariantSet.from_contents(
                turn_id=message.turn_id or message.id,
                contents=[base, new_content],
                selected_index=1,
            )
        else:
            message.variants.variants.append(ConsoleVariant(content=new_content))
            message.variants.selected_index = len(message.variants.variants) - 1
        message.content = message.variants.current.content
        message.status = "complete"
        self._bump_message_speech_revision(message.id)
        self._persist_existing_message(message)
        return self._snapshot(message)

    def select_variant(
        self, message_id: str, selected_index: int
    ) -> ConsoleChatMessage:
        """Select one existing variant by index."""
        message = self._message_or_raise(message_id)
        self._materialize_stream_buffer(message)
        if message.variants is None:
            raise ValueError("Message has no variants.")
        if selected_index < 0 or selected_index >= len(message.variants.variants):
            raise ValueError("selected_index must reference an existing variant")
        message.variants.selected_index = selected_index
        message.content = message.variants.current.content
        self._bump_message_speech_revision(message.id)
        self._persist_existing_message(message)
        return self._snapshot(message)

    def persist_session_if_needed(self, session_id: str) -> str | None:
        """Persist a session once, returning its persisted conversation ID.

        Returns:
            The persisted conversation ID, or ``None`` when no persistence
            adapter is configured.

        Raises:
            ValueError: If ``runtime_backend`` is not exactly ``"local"`` or
                ``"server"``.
        """
        session = self._session_or_raise(session_id)
        if session.persisted_conversation_id is not None:
            return session.persisted_conversation_id
        if self.persistence is None:
            return None
        if (
            type(session.runtime_backend) is not str
            or session.runtime_backend not in {"local", "server"}
        ):
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
        session.persisted_conversation_id = self.persistence.create_conversation(
            conversation_title=session.title,
            workspace_id=persisted_workspace_id,
            scope_type=scope_type,
            system_prompt=session.settings.system_prompt
            if session.settings is not None
            else None,
            **identity_kwargs,
        )
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
        session.settings = replace(session.settings, system_prompt=normalized)
        persisted = True
        if (
            session.persisted_conversation_id is not None
            and self.persistence is not None
        ):
            update_system_prompt = getattr(
                self.persistence,
                "update_conversation_system_prompt",
                None,
            )
            if callable(update_system_prompt):
                try:
                    update_system_prompt(
                        conversation_id=session.persisted_conversation_id,
                        system_prompt=normalized,
                    )
                except Exception:
                    persisted = False
                    logger.bind(
                        session_id=session_id,
                        conversation_id=session.persisted_conversation_id,
                    ).exception(
                        "Failed to persist Console session system prompt; "
                        "in-memory session keeps the applied value."
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
        session.settings = replace(session.settings, pinned_prefill=normalized)
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

    def _persist_new_message(
        self,
        *,
        session_id: str,
        message: ConsoleChatMessage,
        citation_write: SealedCitationWrite | None = None,
        force_stable_message_id: bool = False,
        terminal_persistence: bool = False,
    ) -> None:
        if self.persistence is None:
            return
        conversation_id = self.persist_session_if_needed(session_id)
        if conversation_id is None:
            return
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
            # Generation messages (Task 5) pin the DB row to the SAME id as
            # the store's own native tree-node id: ``message.id`` is already
            # a globally-unique uuid4, and ``add_message`` accepts an
            # explicit id. This makes ``persisted_message_id == message.id``
            # for generation messages specifically, so the narrow
            # keep/append-variant ops -- which callers address by the
            # store's native ``message_id`` -- can pass
            # ``message.persisted_message_id`` straight through with no
            # separate id-translation bookkeeping. Every other message kind
            # keeps letting the DB assign its own id (unchanged).
            message_id=message.id
            if message.generation_metadata or force_stable_message_id
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
        if citation_write is not None:
            create_kwargs["citation_write"] = citation_write
        if terminal_persistence:
            persisted_message_id = self._create_terminal_message(
                create_kwargs=create_kwargs,
                citation_write=citation_write,
            )
            if persisted_message_id is None:
                self._pending_persistence_message_ids.discard(message.id)
                return
        else:
            persisted_message_id = self.persistence.create_message(**create_kwargs)
        message.persisted_message_id = persisted_message_id
        self._pending_persistence_message_ids.discard(message.id)
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

    def _persist_existing_message(
        self,
        message: ConsoleChatMessage,
        *,
        update_feedback: bool = False,
    ) -> None:
        if self.persistence is None:
            return
        if message.persisted_message_id is None:
            self._persist_pending_message_if_ready(message)
            return
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
        self.persistence.update_message_content(**update_kwargs)
        self._enqueue_sync_v2_message_if_ready(message)

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
            restored = replace(message, citation_presentation=None)
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
            restored = replace(node, citation_presentation=None)
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
        self._messages_by_session[session_id] = path

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
