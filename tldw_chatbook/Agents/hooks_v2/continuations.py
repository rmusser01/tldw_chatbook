"""Bounded proposal values; the Console queue remains the admission owner."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .models import HookResult


class ContinuationPolicy:
    """Finite scheduler limits, independent of ordinary run budget checks."""

    @staticmethod
    def permits(
        *,
        admitted_turns: int,
        elapsed_seconds: float,
        foreground_waiting: bool,
        revoked: bool,
        draining: bool,
        closed: bool,
        vetoed: bool,
    ) -> bool:
        return (
            0 <= admitted_turns < 3
            and math.isfinite(elapsed_seconds)
            and 0 <= elapsed_seconds < 120
            and not any((foreground_waiting, revoked, draining, closed, vetoed))
        )


def continuation_key(parent_turn_id: str, event_id: str) -> tuple[str, str]:
    """Return the host's durable identity; hook output never supplies it."""
    return parent_turn_id, event_id


def combine_proposals(proposals: tuple[HookResult, ...]) -> str | None:
    """Keep valid messages whole and stable; refuse overflow and vetoes."""
    messages = []
    for result in proposals:
        if result.stop_continuations or result.decision == "deny":
            return None
        if result.continuation is None:
            continue
        message = result.continuation.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        if len(message.encode("utf-8")) > 4096:
            return None
        messages.append(message)
    combined = "\n\n".join(messages)
    if not combined or len(combined.encode("utf-8")) > 8192:
        return None
    return combined


@dataclass(frozen=True, slots=True)
class ContinuationReceipt:
    """Body-free machine lineage written in the existing acceptance transaction."""

    parent_turn_id: str
    stop_event_id: str
    parent_assistant_message_id: str
    chain_id: str
    admitted_turns: int
    initiator: str = "hook_continuation"


class ContinuationAdmissionRefused(PermissionError):
    """The live gate definitively refused before the transaction committed."""


class ContinuationAdmission:
    """One-use live admission, consumed by the existing acceptance transaction."""

    def __init__(
        self,
        *,
        session_id: str,
        entry_id: str,
        receipt: ContinuationReceipt,
        current: Callable[[], bool],
    ) -> None:
        from threading import Lock
        from uuid import uuid4

        self._identity = {
            "gate_id": uuid4().hex,
            "session_id": session_id,
            "entry_id": entry_id,
            "parent_turn_id": receipt.parent_turn_id,
            "stop_event_id": receipt.stop_event_id,
            "chain_id": receipt.chain_id,
        }
        self._lock = Lock()
        self._state = "pending"
        self._current = current

    def invalidate(self) -> None:
        """Synchronously revoke only pending admission; never reopen consumption."""
        with self._lock:
            if self._state == "pending":
                self._state = "invalidated"

    def consume(self) -> None:
        """Linearize admission, without claiming that the transaction committed."""
        # External authority callbacks must never run while holding the gate lock.
        current = self._current()
        with self._lock:
            if self._state != "pending" or not current:
                self._state = "invalidated" if self._state == "pending" else self._state
                raise ContinuationAdmissionRefused(
                    "hook_continuation_admission_refused"
                )
            self._state = "consumed"

    def durable_acceptance_fingerprint(self) -> dict[str, str]:
        """Bind immutable host identity, never mutable authority or proposal text."""
        return dict(self._identity)

    def write(
        self, *, writer: object, conversation_id: str, message_ids: Mapping[str, str]
    ) -> None:
        """Refusal rolls back the caller-owned messages, checkpoint and receipt."""
        self.consume()
