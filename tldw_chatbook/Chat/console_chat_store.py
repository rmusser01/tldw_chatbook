"""Native Console chat session store and persistence facade."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol
from uuid import uuid4

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleMessageStatus,
    ConsoleWorkspaceContext,
)


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


@dataclass
class ConsoleChatSession:
    """A native Console chat session."""

    title: str = "Chat 1"
    workspace_id: str = "global"
    id: str = field(default_factory=lambda: str(uuid4()))
    persisted_conversation_id: str | None = None


class ConsoleChatStore:
    """Manage native Console sessions and messages before UI integration."""

    def __init__(
        self,
        *,
        persistence: ConsoleChatPersistence | None = None,
        workspace_context: ConsoleWorkspaceContext | None = None,
    ) -> None:
        self.persistence = persistence
        self.workspace_context = workspace_context or ConsoleWorkspaceContext()
        self.active_session_id: str | None = None
        self._sessions: dict[str, ConsoleChatSession] = {}
        self._messages_by_session: dict[str, list[ConsoleChatMessage]] = {}
        self._message_session_index: dict[str, str] = {}
        self._pending_persistence_message_ids: set[str] = set()

    def ensure_session(
        self,
        *,
        title: str = "Chat 1",
        workspace_id: str | None = None,
    ) -> ConsoleChatSession:
        """Return the active session, creating one when needed."""
        if self.active_session_id is not None:
            return self._sessions[self.active_session_id]
        return self.create_session(title=title, workspace_id=workspace_id)

    def create_session(
        self,
        *,
        title: str = "Chat 1",
        workspace_id: str | None = None,
    ) -> ConsoleChatSession:
        """Create and activate a new native Console session."""
        session = ConsoleChatSession(
            title=title,
            workspace_id=workspace_id or self.workspace_context.active_workspace_id,
        )
        self._sessions[session.id] = session
        self._messages_by_session[session.id] = []
        self.active_session_id = session.id
        return session

    def switch_session(self, session_id: str) -> ConsoleChatSession:
        """Activate an existing session."""
        session = self._session_or_raise(session_id)
        self.active_session_id = session.id
        return session

    def set_workspace_context(self, workspace_context: ConsoleWorkspaceContext) -> None:
        """Replace the active workspace context."""
        self.workspace_context = workspace_context

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
        return [self._snapshot(message) for message in self._messages_by_session[session_id]]

    def get_message(self, message_id: str) -> ConsoleChatMessage:
        """Return a message by native message ID."""
        return self._snapshot(self._message_or_raise(message_id))

    def append_stream_chunk(self, message_id: str, chunk: str) -> ConsoleChatMessage:
        """Append streamed assistant content to an existing message."""
        message = self._message_or_raise(message_id)
        self._validate_can_stream(message)
        message.content += chunk
        message.status = "streaming"
        self._persist_pending_message_if_ready(message)
        return self._snapshot(message)

    def mark_message_complete(self, message_id: str) -> ConsoleChatMessage:
        """Mark a message complete and flush final visible content to persistence."""
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
        message.status = "complete"
        self._persist_existing_message(message)
        return self._snapshot(message)

    def mark_message_stopped(self, message_id: str) -> ConsoleChatMessage:
        """Mark a message stopped and flush final visible content to persistence."""
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
        message.status = "stopped"
        self._persist_existing_message(message)
        return self._snapshot(message)

    def mark_message_failed(self, message_id: str) -> ConsoleChatMessage:
        """Mark a message failed and flush final visible content to persistence."""
        message = self._message_or_raise(message_id)
        self._validate_can_mark_terminal(message)
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
            feedback=None,
        )
        self._pending_persistence_message_ids.discard(message.id)

    def _persist_existing_message(self, message: ConsoleChatMessage) -> None:
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
            feedback=None,
            update_parent=False,
            update_feedback=False,
        )

    def _persist_pending_message_if_ready(self, message: ConsoleChatMessage) -> None:
        if (
            self.persistence is None
            or message.id not in self._pending_persistence_message_ids
            or not message.content
        ):
            return
        session_id = self._message_session_index[message.id]
        self._persist_new_message(session_id=session_id, message=message)

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

    @staticmethod
    def _snapshot(message: ConsoleChatMessage) -> ConsoleChatMessage:
        return replace(message)

    @staticmethod
    def _persistence_scope(session: ConsoleChatSession) -> tuple[str, str | None]:
        if session.workspace_id and session.workspace_id != "global":
            return "workspace", session.workspace_id
        return "global", None

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
