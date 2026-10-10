"""Same-store admission for a Console request before complete preparation."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterator
from weakref import ReferenceType, ref

from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
    _validate_identifier,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleChatSession,
        ConsoleChatStore,
    )


@dataclass(frozen=True, slots=True, eq=False)
class ConsoleReceivedTurnClaim:
    """Opaque admission identity; it contains no submitted body or authority."""

    session_id: str
    request_id: str
    generation: int
    origin: ConsoleSubmissionOrigin
    _store_ref: ReferenceType[ConsoleChatStore] = field(repr=False)
    _session_ref: ReferenceType[ConsoleChatSession] = field(repr=False)
    _session_incarnation: str = field(repr=False)
    _binding_revision: int = field(repr=False)
    _ephemeral: bool = field(repr=False)
    _sealed: bool = field(default=False, repr=False)
    draft_revision: int | None = None

    @property
    def sealed(self) -> bool:
        """Whether its store irrevocably stopped this received owner."""
        return self._sealed


def _received_turn_matches_session_locked(
    store: ConsoleChatStore, claim: ConsoleReceivedTurnClaim
) -> bool:
    """Check original recovery provenance without granting current admission."""
    session = store._sessions.get(claim.session_id)
    return (
        claim._store_ref() is store
        and session is not None
        and claim._session_ref() is session
        and session.incarnation_id == claim._session_incarnation
        and session.conversation_binding_revision == claim._binding_revision
        and session.ephemeral is claim._ephemeral
    )


def _received_turn_is_current_locked(
    store: ConsoleChatStore, claim: ConsoleReceivedTurnClaim
) -> bool:
    """Check exact slot and source fields while its admission lock is held."""
    return (
        _received_turn_matches_session_locked(store, claim)
        and store._preparations_by_session.get(claim.session_id) is claim
        and not claim._sealed
    )


class ConsoleReceivedTurnAdmissionMixin:
    """Focused methods using the store's existing lock and admission slot."""

    def claim_received_turn(
        self: ConsoleChatStore,
        session_id: str,
        request_id: str,
        *,
        origin: ConsoleSubmissionOrigin = ConsoleSubmissionOrigin.MANUAL,
        draft_revision: int | None = None,
        _allow_draft_change: bool = False,
    ) -> ConsoleReceivedTurnClaim | None:
        """Reserve one live session before moving complete-request inputs.

        Returns:
            An exact received owner, or None when the live slot is unavailable.

        Raises:
            TypeError: If origin is not the existing Console origin enum.
            ValueError: If the request identifier is malformed.
        """
        _validate_identifier(request_id, "received request ID")
        if type(origin) is not ConsoleSubmissionOrigin:
            raise TypeError("origin must be ConsoleSubmissionOrigin")
        if draft_revision is not None and (
            not isinstance(draft_revision, int)
            or isinstance(draft_revision, bool)
            or draft_revision < 0
        ):
            raise ValueError("draft_revision must be a non-negative integer")
        if not isinstance(session_id, str) or not session_id:
            return None
        with self._preparation_lock:
            session = self._sessions.get(session_id)
            if session is None or (
                draft_revision is not None
                and not _allow_draft_change
                and session.draft_revision != draft_revision
            ):
                return None
            current = self._preparations_by_session.get(session_id)
            if isinstance(current, ConsoleReceivedTurnClaim):
                return None
            if current is not None and current.state not in {
                ConsoleTurnPreparationState.CANCELLED,
                ConsoleTurnPreparationState.SETTLED,
            }:
                return None
            self._received_turn_sequence += 1
            claim = ConsoleReceivedTurnClaim(
                session_id=session_id,
                request_id=request_id,
                generation=self._received_turn_sequence,
                origin=origin,
                draft_revision=draft_revision,
                _store_ref=ref(self),
                _session_ref=ref(session),
                _session_incarnation=session.incarnation_id,
                _binding_revision=session.conversation_binding_revision,
                _ephemeral=session.ephemeral,
            )
            self._preparations_by_session[session_id] = claim
            return claim

    def received_turn_for_session(
        self: ConsoleChatStore, session_id: str | None
    ) -> ConsoleReceivedTurnClaim | None:
        """Return the received slot occupant, including a sealed owner."""
        if not isinstance(session_id, str) or not session_id:
            return None
        with self._preparation_lock:
            current = self._preparations_by_session.get(session_id)
            return current if isinstance(current, ConsoleReceivedTurnClaim) else None

    def received_turn_matches_session(
        self: ConsoleChatStore, claim: ConsoleReceivedTurnClaim
    ) -> bool:
        """Validate a retired claim's original input-recovery source only."""
        if type(claim) is not ConsoleReceivedTurnClaim:
            return False
        with self._preparation_lock:
            return _received_turn_matches_session_locked(self, claim)

    def received_turn_is_current(
        self: ConsoleChatStore, claim: ConsoleReceivedTurnClaim
    ) -> bool:
        """Validate an exact received owner before routes without full preparation."""
        if type(claim) is not ConsoleReceivedTurnClaim:
            return False
        with self._preparation_lock:
            return _received_turn_is_current_locked(self, claim)

    def promote_received_turn(
        self: ConsoleChatStore,
        claim: ConsoleReceivedTurnClaim,
        preparation: ConsoleTurnPreparation,
    ) -> ConsoleTurnPreparation | None:
        """Replace an exact current receipt with a complete preparation atomically.

        Returns:
            The preparation on a promotion win, otherwise None.

        Raises:
            TypeError: If preparation is not a complete preparation value.
        """
        if not isinstance(preparation, ConsoleTurnPreparation):
            raise TypeError("preparation must be ConsoleTurnPreparation")
        if type(claim) is not ConsoleReceivedTurnClaim:
            return None
        with self._preparation_lock:
            if (
                not _received_turn_is_current_locked(self, claim)
                or preparation.session_id != claim.session_id
                or preparation.ephemeral is not claim._ephemeral
                or preparation.preparation_id in self._preparations_by_id
                or preparation.preparation_id in self._durable_tombstones
            ):
                return None
            if claim.draft_revision is not None:
                preparation = replace(
                    preparation, input_draft_revision=claim.draft_revision
                )
            self._preparations_by_session[claim.session_id] = preparation
            self._preparations_by_id[preparation.preparation_id] = preparation
            return preparation

    def seal_received_turn(
        self: ConsoleChatStore, claim: ConsoleReceivedTurnClaim
    ) -> bool:
        """Stop exact receipt promotion while retaining its occupied slot."""
        if type(claim) is not ConsoleReceivedTurnClaim:
            return False
        with self._preparation_lock:
            if (
                claim._store_ref() is not self
                or self._preparations_by_session.get(claim.session_id) is not claim
            ):
                return False
            object.__setattr__(claim, "_sealed", True)
            return True

    def release_received_turn(
        self: ConsoleChatStore, claim: ConsoleReceivedTurnClaim
    ) -> bool:
        """Retire only an exact received owner; promoted/successor slots survive."""
        if type(claim) is not ConsoleReceivedTurnClaim:
            return False
        with self._preparation_lock:
            if (
                claim._store_ref() is not self
                or self._preparations_by_session.get(claim.session_id) is not claim
            ):
                return False
            object.__setattr__(claim, "_sealed", True)
            self._preparations_by_session.pop(claim.session_id)
            return True


@dataclass(frozen=True, slots=True)
class _ReceivedTurnBinding:
    store: ConsoleChatStore
    claim: ConsoleReceivedTurnClaim
    task: asyncio.Task


_received_turn_binding: ContextVar[_ReceivedTurnBinding | None] = ContextVar(
    "console_received_turn_binding", default=None
)


@contextmanager
def bind_received_turn_claim(
    store: ConsoleChatStore, claim: ConsoleReceivedTurnClaim
) -> Iterator[None]:
    """Bind admission identity around one live submit call on this exact Task."""
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("Received turn requires its owning task.")
    if type(claim) is not ConsoleReceivedTurnClaim or claim._store_ref() is not store:
        raise RuntimeError("Received turn owner changed.")
    token = _received_turn_binding.set(_ReceivedTurnBinding(store, claim, task))
    try:
        yield
    finally:
        _received_turn_binding.reset(token)


def received_turn_claim_for(
    store: ConsoleChatStore, session_id: str
) -> ConsoleReceivedTurnClaim | None:
    """Read a same-Task binding without converting stale ownership to admission.

    A child Task inherits ContextVars but cannot borrow the parent's claim.
    Exact bound keys return even a sealed/released claim so promotion refuses
    authoritatively instead of falling back to ordinary admission.

    Raises:
        RuntimeError: If the owning Task attempts another store or session.
    """
    binding = _received_turn_binding.get()
    if binding is None:
        return None
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    if binding.task is not task:
        return None
    if binding.store is not store or binding.claim.session_id != session_id:
        raise RuntimeError("Received turn owner changed.")
    return binding.claim
