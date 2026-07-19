"""Native Console chat session store and persistence facade."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Protocol, Sequence
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
        """
        self.persistence = persistence
        self.workspace_context = workspace_context or ConsoleWorkspaceContext()
        self.sync_v2_chat_producer = sync_v2_chat_producer
        self.sync_v2_server_profile_id = sync_v2_server_profile_id
        self.sync_v2_authenticated_principal_id = sync_v2_authenticated_principal_id
        self.sync_v2_workspace_scope = sync_v2_workspace_scope
        self.active_session_id: str | None = None
        self._sessions: dict[str, ConsoleChatSession] = {}
        self._messages_by_session: dict[str, list[ConsoleChatMessage]] = {}
        self._message_session_index: dict[str, str] = {}
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
        return self.create_session(title=title, workspace_id=workspace_id, settings=settings)

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
        self.active_session_id = session.id
        return session

    def restore_persisted_session(
        self,
        *,
        title: str,
        workspace_id: str | None,
        persisted_conversation_id: str,
        messages: Iterable[ConsoleChatMessage],
        settings: ConsoleSessionSettings | None = None,
    ) -> ConsoleChatSession:
        """Create and activate a native session from persisted conversation data.

        Args:
            title: Display title for the restored Console session.
            workspace_id: Workspace scope recorded on the persisted conversation,
                or ``None`` to use the current store workspace context.
            persisted_conversation_id: Durable Chat conversation identifier.
            messages: Native Console messages reconstructed from persisted data.
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
        restored_messages: list[ConsoleChatMessage] = []
        for message in messages:
            restored = replace(message)
            restored_messages.append(restored)
            self._message_session_index[restored.id] = session.id
        self._messages_by_session[session.id] = restored_messages
        return session

    def switch_session(self, session_id: str) -> ConsoleChatSession:
        """Activate an existing session."""
        session = self._session_or_raise(session_id)
        self.active_session_id = session.id
        return session

    def rename_session(self, session_id: str, title: str) -> ConsoleChatSession:
        """Rename an existing native Console session."""
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("Console chat session title cannot be blank.")
        session = self._session_or_raise(session_id)
        session.title = normalized_title
        return session

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

        for message in self._messages_by_session.get(session_id, []):
            self._message_session_index.pop(message.id, None)
            self._stream_chunks_by_message.pop(message.id, None)
            self._stream_materialized_counts.pop(message.id, None)
            self._pending_persistence_message_ids.discard(message.id)

        self._messages_by_session.pop(session_id, None)
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

        messages_by_session = messages_by_session or {}
        for session in restored_sessions:
            self._sessions[session.id] = replace(session)
            restored_messages: list[ConsoleChatMessage] = []
            for message in messages_by_session.get(session.id, ()):
                restored_message = replace(message)
                restored_messages.append(restored_message)
                self._message_session_index[restored_message.id] = session.id
            self._messages_by_session[session.id] = restored_messages

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
        self._messages_by_session[session_id].append(message)
        self._sessions[session_id].updated_at = _utc_now_iso()
        self._message_session_index[message.id] = session_id
        if persist:
            self._persist_new_message_or_defer(session_id=session_id, message=message)
        return self._snapshot(message)

    def messages_for_session(self, session_id: str) -> list[ConsoleChatMessage]:
        """Return messages for a session in transcript order."""
        self._session_or_raise(session_id)
        for message in self._messages_by_session[session_id]:
            self._materialize_stream_buffer(message)
        return [self._snapshot(message) for message in self._messages_by_session[session_id]]

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

    def update_message_content(self, message_id: str, content: str) -> ConsoleChatMessage:
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
            raise ValueError("Wait for response to finish before deleting this message.")
        session_id = self._message_session_index.pop(message_id)
        messages = self._messages_by_session[session_id]
        self._messages_by_session[session_id] = [
            candidate for candidate in messages if candidate.id != message_id
        ]
        self._stream_chunks_by_message.pop(message_id, None)
        self._stream_materialized_counts.pop(message_id, None)
        self._pending_persistence_message_ids.discard(message_id)
        self._variant_stream_bases.pop(message_id, None)
        return self._snapshot(message)

    def session_id_for_message(self, message_id: str) -> str:
        """Return the owning session ID for a message."""
        if message_id not in self._message_session_index:
            raise KeyError(f"Unknown Console message: {message_id}")
        return self._message_session_index[message_id]

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

    def prepare_message_retry(self, message_id: str) -> ConsoleChatMessage:
        """Prepare a failed assistant message to receive replacement stream content."""
        message = self._message_or_raise(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can be retried.")
        if message.status != "failed":
            raise ValueError(f"Only failed messages can be retried, not {message.status}.")
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

    def select_variant(self, message_id: str, selected_index: int) -> ConsoleChatMessage:
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
        session.persisted_conversation_id = self.persistence.create_conversation(
            assistant_kind="generic",
            assistant_id="console",
            conversation_title=session.title,
            workspace_id=persisted_workspace_id,
            scope_type=scope_type,
            system_prompt=session.settings.system_prompt if session.settings is not None else None,
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
            needed (session not yet saved, or no persistence configured).
        """
        session = self._session_or_raise(session_id)
        normalized = system_prompt if isinstance(system_prompt, str) and system_prompt.strip() else None
        if session.settings is not None:
            session.settings = replace(session.settings, system_prompt=normalized)
        persisted = True
        if session.persisted_conversation_id is not None and self.persistence is not None:
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

    def _persist_new_message_or_defer(self, *, session_id: str, message: ConsoleChatMessage) -> None:
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

    def _persist_new_message(self, *, session_id: str, message: ConsoleChatMessage) -> None:
        if self.persistence is None:
            return
        conversation_id = self.persist_session_if_needed(session_id)
        if conversation_id is None:
            return
        create_kwargs: dict[str, Any] = dict(
            conversation_id=conversation_id,
            sender=message.role.value,
            content=message.content,
            message_id=None,
            parent_message_id=None,
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
        conversation_id = session.persisted_conversation_id if session is not None else None
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
        session_id = self._message_session_index.get(message.id)
        if session_id is None:
            return None
        previous: str | None = None
        for candidate in self._messages_by_session.get(session_id, []):
            if candidate.id == message.id:
                return previous
            if candidate.persisted_message_id is not None:
                previous = candidate.persisted_message_id
        return None

    @staticmethod
    def _sync_variant_metadata(message: ConsoleChatMessage) -> dict[str, str | int | None]:
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

    def _record_sync_v2_message_version(self, stable_key: str, result: dict[str, Any]) -> None:
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
        session_id = self._message_session_index.get(message_id)
        if session_id is None:
            raise KeyError(f"Unknown Console message: {message_id}")
        for message in self._messages_by_session[session_id]:
            if message.id == message_id:
                return message
        raise KeyError(f"Unknown Console message: {message_id}")

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
            raise ValueError(f"Cannot append stream chunks to a {message.status} message.")

    @staticmethod
    def _validate_can_mark_terminal(message: ConsoleChatMessage) -> None:
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can enter terminal stream states.")
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
