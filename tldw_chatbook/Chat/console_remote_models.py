"""Opaque identities for routing to an existing Console execution owner."""

import asyncio
import re
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, field_validator

from tldw_chatbook.Canvas.limits import validate_opaque_identifier


class ConsoleHostIdentity(BaseModel):
    """One retained child, scoped to its parent server lifetime."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    server_epoch: str
    console_id: str

    @field_validator("*", mode="before")
    @classmethod
    def validate_identity(cls, value: object) -> object:
        """Reject paths/display labels as well as malformed opaque identities."""
        if value is not None:
            validate_opaque_identifier(value)
            if re.fullmatch(r"[A-Za-z0-9_:.-]+", value) is None or value in {".", ".."}:
                raise ValueError("identity must be opaque")
        return value


class ConsoleTarget(ConsoleHostIdentity):
    """Exact live chat incarnation and optional native work identities."""

    session_id: str
    incarnation_id: str
    turn_id: str | None = None
    run_id: str | None = None
    handle_id: str | None = None
    round_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConsoleEventBatch:
    """A bounded projection read and its snapshot recovery boundary."""

    sequence: "int"
    reset_required: "bool"
    events: "tuple[dict, ...]"


class ConsoleEventBuffer:
    """Keep bounded history and a sticky reset, using one shared wake signal.

    Access belongs to the serving loop. A snapshot acknowledges only the
    current sequence; a stale snapshot cannot erase a concurrent loss.
    """

    def __init__(self, limit: "int" = 64) -> "None":
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise ValueError("invalid event-buffer limit")
        self._events: deque[dict] = deque(maxlen=limit)
        self._sequence = 0
        self._reset_required = False
        self._discarded_through = 0
        self._changed = asyncio.Event()

    def publish(self, kind: "str", payload: "dict") -> "int":
        """Retain a projection, marking any lost history until snapshot ack."""
        self._sequence += 1
        if len(self._events) == self._events.maxlen:
            self._discarded_through = self._events[0]["sequence"]
            self._reset_required = True
        if payload.get("reset_required") is True:
            self._reset_required = True
        self._events.append(
            {"sequence": self._sequence, "kind": kind, "payload": deepcopy(payload)}
        )
        return self._sequence

    def read(self, after_sequence: "int") -> "ConsoleEventBatch":
        """Read without consuming another observer's history or reset marker."""
        gap = after_sequence < self._discarded_through
        return ConsoleEventBatch(
            self._sequence,
            self._reset_required or gap or after_sequence > self._sequence,
            tuple(
                deepcopy(event)
                for event in self._events
                if event["sequence"] > after_sequence
            ),
        )

    async def wait_changed(self, after_sequence: "int") -> "None":
        """Wait without queues, per-event tasks or a check/clear lost wake."""
        while self._sequence <= after_sequence:
            self._changed.clear()
            await self._changed.wait()

    def note_changed(self, sequence: "int") -> "None":
        """Wake every observer with a single bounded signal."""
        if sequence <= self._sequence:
            self._changed.set()

    def acknowledge_snapshot(self, sequence: "int") -> "None":
        """Clear loss only when a current authoritative snapshot reconciles it."""
        if sequence == self._sequence:
            self._reset_required = False
            self._discarded_through = sequence
            self._events.clear()
