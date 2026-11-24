"""Event-loop-local admission for a single audio output."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from threading import Event

_Retire = Callable[[], Awaitable[bool]]


@dataclass(frozen=True, eq=False)
class AudioOutputLease:
    """One host-issued start authority for a particular output generation."""

    generation: int
    _host: AudioOutputOwnership
    _owner: str
    _revoked: Event = field(default_factory=Event, init=False, repr=False)

    @property
    def revoked(self) -> bool:
        """Return the one-way, thread-visible cancellation signal."""
        return self._revoked.is_set()

    def revoke(self) -> None:
        """Fence the exact held lease on the owner loop without releasing it."""
        if self._host._owns(self):
            self._revoked.set()

    def is_current(self) -> bool:
        """Return whether this exact lease can still start or publish output."""
        return (
            not self.revoked
            and self._host._owns(self)
            and self._host._generation == self.generation
        )

    def release(self) -> None:
        """Acknowledge that this lease's exact output has physically retired."""
        if self._host._owns(self):
            self._revoked.set()
            self._host._slot = None
            self._host._blocked = False


class AudioOutputOwnership:
    """Coordinate output starts and exact-owner retirement on the app's event loop."""

    def __init__(self) -> None:
        self._slot: tuple[AudioOutputLease, _Retire] | None = None
        self._generation = 0
        self._blocked = False
        self._retirement: asyncio.Task[None] | None = None

    @property
    def owner(self) -> str | None:
        """Return the diagnostic label of the output still in custody."""
        return self._slot[0]._owner if self._slot is not None else None

    @property
    def blocked(self) -> bool:
        """Return whether physical retirement failed for the retained output."""
        return self._blocked

    def _owns(self, lease: AudioOutputLease) -> bool:
        return self._slot is not None and self._slot[0] is lease

    async def _retire_owned(
        self, lease: AudioOutputLease, retire: _Retire, published: asyncio.Event
    ) -> None:
        # An eager task can enter here before create_task returns to acquire.
        await published.wait()
        try:
            confirmed = await retire()
        except (Exception, asyncio.CancelledError):
            confirmed = False
        if confirmed is True:
            lease.release()
        elif self._owns(lease):
            self._blocked = True

    async def acquire(self, owner: str, retire: _Retire) -> AudioOutputLease:
        """Revoke earlier starts and wait for the previous output to retire.

        Args:
            owner: Diagnostic label for this start.
            retire: Callback that retires this start's exact resource and returns
                True after confirming it stopped. Move blocking work off-loop.

        Returns:
            The lease authorizing the newest start on the app's event loop.

        Raises:
            RuntimeError: Retirement failed or a newer request superseded this one.
            asyncio.CancelledError: The caller cancelled its request.
        """
        self._generation += 1
        generation = self._generation
        if self._slot is not None:
            self._slot[0].revoke()
        if self._retirement is None and self._slot is not None:
            published = asyncio.Event()
            self._retirement = asyncio.create_task(
                self._retire_owned(*self._slot, published)
            )
            published.set()
        if self._retirement is not None:
            # The coordinator retains this work even if every waiting start leaves.
            await asyncio.shield(self._retirement)
            if generation != self._generation:
                raise RuntimeError("Audio output request was superseded")
            if self._blocked:
                raise RuntimeError("Audio output retirement is blocked")
        self._retirement = None
        lease = AudioOutputLease(generation, self, owner)
        self._slot = lease, retire
        return lease
