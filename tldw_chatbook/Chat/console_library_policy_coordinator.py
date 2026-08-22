"""In-process publication and execution capture for Library policy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from tldw_chatbook.Chat.console_library_policy import (
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyHolder,
    ConsoleLibraryPolicyReadResult,
    ConsoleLibraryPolicySnapshot,
    ConsoleLibraryPolicyWriteResult,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.Chat.console_library_policy_repository import (
    ConsoleLibraryPolicyRepository,
)


@dataclass(slots=True)
class _RegisteredHolder:
    conversation_id: str | None
    holder: ConsoleLibraryPolicyHolder


class ConsoleLibraryPolicyCoordinator:
    """Own live holders while durable repository work runs off-loop."""

    def __init__(self, repository: ConsoleLibraryPolicyRepository) -> None:
        self.repository = repository
        self._holders: dict[str, _RegisteredHolder] = {}

    def register_holder(
        self,
        session_id: str,
        conversation_id: str | None,
        holder: ConsoleLibraryPolicyHolder,
    ) -> None:
        """Bind one live holder for same-process committed publication."""
        self._holders[session_id] = _RegisteredHolder(conversation_id, holder)

    def unregister_holder(self, session_id: str) -> None:
        """Remove one closed session holder."""
        self._holders.pop(session_id, None)

    async def load(
        self, session_id: str, conversation_id: str
    ) -> ConsoleLibraryPolicyReadResult:
        """Read durable policy off-loop and publish its effective result."""
        registered = self._require_session(session_id)
        registered.conversation_id = conversation_id
        result = await asyncio.to_thread(self.repository.read, conversation_id)
        self._publish(conversation_id, result.snapshot)
        return result

    async def save(
        self,
        session_id: str,
        candidate: ConsoleLibraryPolicyCandidate,
    ) -> ConsoleLibraryPolicyWriteResult:
        """Commit one insert/CAS and publish only the committed snapshot."""
        registered = self._require_session(session_id)
        conversation_id = registered.conversation_id
        if conversation_id is None:
            raise ValueError("A durable conversation is required for policy save.")
        registered.holder.save_pending = True
        try:
            revision = registered.holder.snapshot.policy_revision
            if revision is None:
                result = await asyncio.to_thread(
                    self.repository.insert,
                    conversation_id,
                    candidate,
                )
            else:
                result = await asyncio.to_thread(
                    self.repository.compare_and_swap,
                    conversation_id,
                    revision,
                    candidate,
                )
        finally:
            registered.holder.save_pending = False
        if result.status is ConsoleLibraryPolicyWriteStatus.COMMITTED:
            self._publish(conversation_id, result.snapshot)
        return result

    async def capture_for_execution(
        self, session_id: str
    ) -> ConsoleLibraryPolicySnapshot:
        """Perform the execution-time durable read and return frozen authority."""
        registered = self._require_session(session_id)
        conversation_id = registered.conversation_id
        if conversation_id is None:
            return registered.holder.snapshot
        result = await asyncio.to_thread(self.repository.read, conversation_id)
        self._publish(conversation_id, result.snapshot)
        return result.snapshot

    def _publish(
        self,
        conversation_id: str,
        snapshot: ConsoleLibraryPolicySnapshot,
    ) -> None:
        for registered in tuple(self._holders.values()):
            if registered.conversation_id == conversation_id:
                registered.holder.snapshot = snapshot
                registered.holder.save_pending = False

    def _require_session(self, session_id: str) -> _RegisteredHolder:
        try:
            return self._holders[session_id]
        except KeyError:
            raise KeyError(f"Unknown Console session: {session_id}") from None
