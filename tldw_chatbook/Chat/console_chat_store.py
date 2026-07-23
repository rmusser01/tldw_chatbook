"""Native Console chat session store and persistence facade."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    DEFAULT_CONSOLE_SESSION_TITLE,
    ConsoleChatMessage,
    ConsoleMessageFeedback,
    ConsoleMessageRole,
    ConsoleMessageStatus,
    ConsoleVariant,
    ConsoleVariantSet,
    ConsoleWorkspaceContext,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.rag_scope import RagScope, SessionScopeHolder

#: Maximum number of attachments a Console session may stage before send.
MAX_PENDING_ATTACHMENTS = 5


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
    ) -> str:
        """Create a persisted message and return its ID.

        ``attachments``, when given, covers ALL positions (0..N-1) and is
        authoritative over the scalar ``image_data``/``image_mime_type``
        kwargs; ``None`` leaves the pre-split legacy behavior unchanged.
        Optional: fakes used in tests may omit this parameter entirely.
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


class ConsoleChatSyncProducer(Protocol):
    """Sync v2 producer surface used after durable local Chat writes."""

    def enqueue_chat_message(self, **kwargs: Any) -> dict[str, Any]:
        """Enqueue a Chat message into the Sync v2 local outbox."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    #: When set, this is a character-bound session: it persists with the
    #: character's id, forces the plain-provider path, and restores as a
    #: character session (task-427). ``None`` = a normal Console session.
    character_id: int | None = None
    character_name: str | None = None


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
        self._pending_persistence_message_ids: set[str] = set()
        self._stream_chunks_by_message: dict[str, list[str]] = {}
        self._stream_materialized_counts: dict[str, int] = {}
        self._sync_v2_message_versions: dict[str, str] = {}
        self._variant_stream_bases: dict[str, _VariantStreamBase] = {}

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
    ) -> ConsoleChatSession:
        """Create and activate a new native Console session."""
        session = ConsoleChatSession(
            title=title,
            workspace_id=workspace_id or self.workspace_context.active_workspace_id,
            settings=settings,
        )
        self._sessions[session.id] = session
        self._messages_by_session[session.id] = []
        self._nodes_by_session[session.id] = {}
        self._children_by_parent[session.id] = {}
        self._active_leaf_by_session[session.id] = None
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
        )
        session.persisted_conversation_id = str(persisted_conversation_id)
        self._ingest_full_tree(
            session.id,
            all_nodes,
            active_leaf_persisted_id=active_leaf_persisted_id,
        )
        return session

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
            self._message_session_index.pop(message_id, None)
            self._stream_chunks_by_message.pop(message_id, None)
            self._stream_materialized_counts.pop(message_id, None)
            self._pending_persistence_message_ids.discard(message_id)
            self._variant_stream_bases.pop(message_id, None)
            self._native_parent_by_message.pop(message_id, None)

        self._messages_by_session.pop(session_id, None)
        self._nodes_by_session.pop(session_id, None)
        self._children_by_parent.pop(session_id, None)
        self._active_leaf_by_session.pop(session_id, None)
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
        self._stream_chunks_by_message.clear()
        self._stream_materialized_counts.clear()
        self._sync_v2_message_versions.clear()
        # Pre-existing bug fixed while here: the regenerate base snapshots were
        # never cleared on restore, leaking across a state replacement.
        self._variant_stream_bases.clear()
        self._nodes_by_session.clear()
        self._children_by_parent.clear()
        self._native_parent_by_message.clear()
        self._active_leaf_by_session.clear()

        messages_by_session = messages_by_session or {}
        for session in restored_sessions:
            self._sessions[session.id] = replace(session)
            self._nodes_by_session[session.id] = {}
            self._children_by_parent[session.id] = {}
            self._active_leaf_by_session[session.id] = None
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
        if persist:
            self._persist_new_message_or_defer(session_id=session_id, message=message)
        return self._snapshot(message)

    def create_sibling(
        self,
        anchor_message_id: str,
        *,
        role: ConsoleMessageRole,
        content: str = "",
        persist: bool = False,
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
            nodes.pop(node_id, None)
            children_map.pop(node_id, None)
            self._native_parent_by_message.pop(node_id, None)
            self._message_session_index.pop(node_id, None)
            self._stream_chunks_by_message.pop(node_id, None)
            self._stream_materialized_counts.pop(node_id, None)
            self._pending_persistence_message_ids.discard(node_id)
            self._variant_stream_bases.pop(node_id, None)
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

    def active_path_message_ids(self, session_id: str) -> list[str]:
        """Return native ids along the active path, root -> active leaf."""
        self._session_or_raise(session_id)
        ids: list[str] = []
        current = self._active_leaf_by_session.get(session_id)
        while current is not None:
            ids.append(current)
            current = self._native_parent_by_message.get(current)
        ids.reverse()
        return ids

    def siblings_at(
        self, message_id: str
    ) -> tuple[list[ConsoleChatMessage], int, int]:
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
        return self._snapshot(message)

    def mark_message_complete(self, message_id: str) -> ConsoleChatMessage:
        """Mark a message complete and flush final visible content to persistence."""
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
        self._materialize_stream_buffer(message)
        message.status = "complete"
        self._persist_existing_message(message)
        return self._snapshot(message)

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
        base = self._variant_stream_bases.pop(message.id, None)
        if base is not None:
            message.content = base.content
            message.status = base.prior_status
        else:
            message.status = "stopped"
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
        base = self._variant_stream_bases.pop(message.id, None)
        if base is not None:
            message.content = base.content
            message.status = base.prior_status
        else:
            message.status = "failed"
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
        self._persist_existing_message(message)
        return self._snapshot(message)

    def persist_session_if_needed(self, session_id: str) -> str | None:
        """Persist a session once, returning its persisted conversation ID."""
        session = self._session_or_raise(session_id)
        if session.persisted_conversation_id is not None:
            return session.persisted_conversation_id
        if self.persistence is None:
            return None
        scope_type, persisted_workspace_id = self._persistence_scope(session)
        if session.character_id is not None:
            identity_kwargs = {
                "assistant_kind": "character",
                "assistant_id": str(session.character_id),
                "character_id": session.character_id,
                "character_name": session.character_name,
            }
        else:
            identity_kwargs = {
                "assistant_kind": "generic",
                "assistant_id": "console",
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
        if not message.content and not message.attachments:
            self._pending_persistence_message_ids.add(message.id)
            self.persist_session_if_needed(session_id)
            return
        self._persist_new_message(session_id=session_id, message=message)

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
        self, *, session_id: str, message: ConsoleChatMessage
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
            message_id=None,
            parent_message_id=parent_persisted_id,
            feedback=message.feedback,
        )
        # Only engage split addressing when there is something beyond the
        # legacy position-0 slot to address -- a lone image (whether staged
        # via scalar kwargs or a single-item attachments tuple) keeps using
        # the scalar image_data/image_mime_type kwargs exactly as before.
        attachments_payload = None
        if len(message.attachments) > 1:
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
        message.persisted_message_id = self.persistence.create_message(**create_kwargs)
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
        if message.id == self._active_leaf_by_session.get(session_id):
            self._persist_active_leaf(session_id, message.id)
        self._enqueue_sync_v2_message_if_ready(message)

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
            or not message.content
        ):
            return
        session_id = self._message_session_index[message.id]
        self._persist_new_message(session_id=session_id, message=message)

    def _enqueue_sync_v2_message_if_ready(self, message: ConsoleChatMessage) -> None:
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
            restored = replace(message)
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
            restored = replace(node)
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
        root thread. Legacy flat data, by construction, ALWAYS mixes roles at
        the root (every message, both USER and ASSISTANT, was written with
        ``parent_message_id=NULL``), whereas a genuine root-level fork's
        siblings are ALWAYS all USER (an ASSISTANT node's native parent is
        never ``None`` -- it always replies to a user turn, even the very
        first one). So role-homogeneity is the distinguishing signal: when
        every root shares one role, this is a genuine Phase-B branch and must
        be left alone (chaining it would silently splice the newer branch
        onto the older one as a fake parent-child link, corrupting the tree
        so a swipe/resume shows the wrong content); only a role-MIXED root
        set is unambiguously legacy data.

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
        root_roles = {nodes[root_id].role for root_id in roots if root_id in nodes}
        if len(root_roles) <= 1:
            # Every root-level node shares one role: a genuine Phase-B
            # root-level branch (all USER), not legacy flat data (which
            # always mixes USER and ASSISTANT rows at the root). Leave each
            # root independently navigable via `siblings_at`/`set_active_leaf`.
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
        """
        nodes = self._nodes_by_session.get(session_id, {})
        children = self._children_by_parent.get(session_id, {})
        path_ids: list[str] = []
        current = self._active_leaf_by_session.get(session_id)
        while current is not None:
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
        self, session_id: str, message_id: str | None
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
            logger.bind(
                session_id=session_id,
                conversation_id=conversation_id,
            ).exception(
                "Failed to persist Console active-leaf pointer; the in-memory "
                "pointer keeps the applied value."
            )

    def _materialize_stream_buffer(self, message: ConsoleChatMessage) -> None:
        """Fold buffered stream chunks into ``message.content`` if any are new.

        TASK-259: after joining, the chunk list is collapsed to the single
        joined string (in place, preserving any outstanding list references),
        so the next materialize joins only the chunks that arrived since this
        one instead of re-walking the whole stream history on every 0.2s
        tick. The invariant ``"".join(buffer) == full streamed content`` is
        preserved for every reader.

        Args:
            message: Store-owned message whose visible content should
                reflect all chunks appended so far.
        """
        buffer = self._stream_chunks_by_message.get(message.id)
        if not buffer:
            return
        if self._stream_materialized_counts.get(message.id) == len(buffer):
            return
        message.content = "".join(buffer)
        buffer[:] = [message.content]
        self._stream_materialized_counts[message.id] = 1
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
