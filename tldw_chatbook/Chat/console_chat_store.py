"""Native Console chat session store and persistence facade."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Protocol
from uuid import uuid4

from loguru import logger

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
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings


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
    ) -> str:
        """Create a persisted message and return its ID."""

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
    ) -> bool:
        """Update persisted message content."""


class ConsoleChatSyncProducer(Protocol):
    """Sync v2 producer surface used after durable local Chat writes."""

    def enqueue_chat_message(self, **kwargs: Any) -> dict[str, Any]:
        """Enqueue a Chat message into the Sync v2 local outbox."""


@dataclass
class ConsoleChatSession:
    """A native Console chat session."""

    title: str = DEFAULT_CONSOLE_SESSION_TITLE
    workspace_id: str = CONSOLE_GLOBAL_WORKSPACE_ID
    id: str = field(default_factory=lambda: str(uuid4()))
    persisted_conversation_id: str | None = None
    settings: ConsoleSessionSettings | None = None
    draft: str = ""


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

    def append_message(
        self,
        session_id: str,
        *,
        role: ConsoleMessageRole,
        content: str,
        persist: bool = False,
    ) -> ConsoleChatMessage:
        """Append a message to a session and optionally persist it."""
        self._session_or_raise(session_id)
        message = ConsoleChatMessage(
            role=role,
            content=content,
            status=self._initial_status(role=role, content=content),
        )
        self._messages_by_session[session_id].append(message)
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
        return self._snapshot(message)

    def session_id_for_message(self, message_id: str) -> str:
        """Return the owning session ID for a message."""
        if message_id not in self._message_session_index:
            raise KeyError(f"Unknown Console message: {message_id}")
        return self._message_session_index[message_id]

    def append_stream_chunk(self, message_id: str, chunk: str) -> ConsoleChatMessage:
        """Append streamed assistant content to an existing message."""
        message = self._message_or_raise(message_id)
        self._validate_can_stream(message)
        buffer = self._stream_chunks_by_message.setdefault(
            message.id,
            [message.content] if message.content else [],
        )
        buffer.append(chunk)
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
        """Mark a message stopped and flush final visible content to persistence."""
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
        self._materialize_stream_buffer(message)
        message.status = "stopped"
        self._persist_existing_message(message)
        return self._snapshot(message)

    def mark_message_failed(self, message_id: str) -> ConsoleChatMessage:
        """Mark a message failed and flush final visible content to persistence."""
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
        self._materialize_stream_buffer(message)
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
        )
        return session.persisted_conversation_id

    def _persist_new_message_or_defer(self, *, session_id: str, message: ConsoleChatMessage) -> None:
        if self.persistence is None:
            return
        if not message.content:
            self._pending_persistence_message_ids.add(message.id)
            self.persist_session_if_needed(session_id)
            return
        self._persist_new_message(session_id=session_id, message=message)

    def _persist_new_message(self, *, session_id: str, message: ConsoleChatMessage) -> None:
        if self.persistence is None:
            return
        conversation_id = self.persist_session_if_needed(session_id)
        if conversation_id is None:
            return
        message.persisted_message_id = self.persistence.create_message(
            conversation_id=conversation_id,
            sender=message.role.value,
            content=message.content,
            image_data=None,
            image_mime_type=None,
            message_id=None,
            parent_message_id=None,
            feedback=message.feedback,
        )
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
        self.persistence.update_message_content(
            message_id=message.persisted_message_id,
            content=message.content,
            image_data=None,
            image_mime_type=None,
            parent_message_id=None,
            feedback=message.feedback,
            update_parent=False,
            update_feedback=update_feedback,
        )
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
        buffer = self._stream_chunks_by_message.get(message.id)
        if not buffer:
            return
        chunk_count = len(buffer)
        if self._stream_materialized_counts.get(message.id) == chunk_count:
            return
        message.content = "".join(buffer)
        self._stream_materialized_counts[message.id] = chunk_count
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
