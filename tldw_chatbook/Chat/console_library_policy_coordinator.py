"""In-process publication and execution capture for Library policy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from tldw_chatbook.Chat.console_library_policy import (
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyHolder,
    ConsoleLibraryPolicyReadResult,
    ConsoleLibraryPolicySnapshot,
    ConsoleLibraryPolicyWriteResult,
    ConsoleLibraryPolicyWriteStatus,
    normalize_policy_read,
)
from tldw_chatbook.Chat.console_library_policy_repository import (
    ConsoleLibraryPolicyRepository,
)

T = TypeVar("T")


@dataclass(slots=True)
class _RegisteredHolder:
    conversation_id: str | None
    holder: ConsoleLibraryPolicyHolder
    generation: int


class ConsoleLibraryPolicyCoordinator:
    """Own live holders while durable repository work runs off-loop."""

    def __init__(self, repository: ConsoleLibraryPolicyRepository) -> None:
        self.repository = repository
        self._holders: dict[str, _RegisteredHolder] = {}
        self._next_generation = 0

    def register_holder(
        self,
        session_id: str,
        conversation_id: str | None,
        holder: ConsoleLibraryPolicyHolder,
    ) -> None:
        """Bind one live holder for same-process committed publication."""
        self._next_generation += 1
        self._holders[session_id] = _RegisteredHolder(
            conversation_id,
            holder,
            self._next_generation,
        )

    def unregister_holder(self, session_id: str) -> None:
        """Remove one closed session holder."""
        self._holders.pop(session_id, None)

    async def _run_repository_call(
        self, callback: Callable[..., T], /, *args: Any
    ) -> T:
        """Run repository work without splitting an in-memory database.

        ``CharactersRAGDB`` owns one connection per thread. File-backed
        databases therefore belong in ``to_thread``, while ``:memory:`` would
        open a different, empty database on the worker thread. In-memory
        calls stay on their owning event-loop thread; they cannot perform disk
        I/O and are used only by bounded test or ephemeral stores.
        """

        db = getattr(self.repository, "db", None)
        if getattr(db, "is_memory_db", None) is True:
            return callback(*args)
        return await asyncio.to_thread(callback, *args)

    async def load(
        self, session_id: str, conversation_id: str
    ) -> ConsoleLibraryPolicyReadResult:
        """Read durable policy off-loop and publish its effective result."""
        registered = self._require_session(session_id)
        registered.conversation_id = conversation_id
        self._next_generation += 1
        registered.generation = self._next_generation
        return await self._read_current_binding(session_id)

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
                result = await self._run_repository_call(
                    self.repository.insert, conversation_id, candidate
                )
            else:
                result = await self._run_repository_call(
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
        if registered.conversation_id is None:
            return registered.holder.snapshot
        result = await self._read_current_binding(session_id)
        return result.snapshot

    async def _read_current_binding(
        self,
        session_id: str,
    ) -> ConsoleLibraryPolicyReadResult:
        for _attempt in range(2):
            registered = self._require_session(session_id)
            conversation_id = registered.conversation_id
            generation = registered.generation
            if conversation_id is None:
                return normalize_policy_read(None)
            result = await self._run_repository_call(
                self.repository.read, conversation_id
            )
            current = self._holders.get(session_id)
            if (
                current is registered
                and current.generation == generation
                and current.conversation_id == conversation_id
            ):
                self._publish(conversation_id, result.snapshot)
                return result
        result = normalize_policy_read(RuntimeError("session_binding_changed"))
        current = self._holders.get(session_id)
        if current is not None and current.conversation_id is not None:
            self._publish(current.conversation_id, result.snapshot)
        return result

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
