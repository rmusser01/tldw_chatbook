"""Event-loop-local admission for a single audio output."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from threading import Event
from types import MethodType
from weakref import WeakSet

_Retire = Callable[[], Awaitable[bool]]


class OutputSuperseded(RuntimeError):
    """A newer action superseded this request or its intent was cancelled."""


class OutputClosed(RuntimeError):
    """The app has closed audio admission for shutdown."""


class OutputBlocked(RuntimeError):
    """The retained owner could not confirm physical retirement."""

    def __init__(self, owner: str) -> None:
        self.owner = owner
        super().__init__(f"Audio output retirement is blocked for {owner}")


class OutputHandoffRequired(RuntimeError):
    """External playback needs user acknowledgement for this exact epoch."""

    def __init__(self, epoch: object) -> None:
        self.epoch = epoch
        super().__init__("Confirm external playback has stopped before starting audio")


@dataclass(frozen=True, eq=False)
class AudioOutputIntent:
    """One loop-issued user action, reserved before preparing its audio."""

    serial: int
    _host: AudioOutputOwnership

    def is_current(self) -> bool:
        """Return whether this exact intent is eligible for admission."""
        return self._host._intent is self

    def cancel(self) -> None:
        """Discard only this action, preserving the active output."""
        self._host._tentative.discard(self)
        if self.is_current():
            self._host._intent = None

    def promote(self) -> bool:
        """Make an audible probe eligible if no newer audible action intervened."""
        if self not in self._host._tentative:
            return self.is_current()
        self._host._tentative.remove(self)
        if self._host.closing or self.serial <= self._host._eligible_serial:
            return False
        self._host._intent = self
        self._host._eligible_serial = self.serial
        return True


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
        return not self.revoked and self._host._owns(self)

    def authorizes(self, retire_callback: _Retire) -> bool:
        """Check on the owner loop whether this lease may start this exact run."""
        slot = self._host._slot
        if slot is None or slot[0] is not self or self.revoked:
            return False
        registered = slot[1]
        # A fresh attribute lookup creates a new Python bound method object.
        return registered is retire_callback or (
            isinstance(registered, MethodType)
            and isinstance(retire_callback, MethodType)
            and registered.__self__ is retire_callback.__self__
            and registered.__func__ is retire_callback.__func__
        )

    def release(self) -> None:
        """Acknowledge that this lease's exact output has physically retired."""
        if self._host._owns(self):
            self._revoked.set()
            self._host._slot = None
            self._host._blocked = False

    def record_handoff(self) -> None:
        """Record OS acceptance on the loop while this exact launch is in custody."""
        if self._host._owns(self):
            self._host._handoff_epoch = object()


class AudioOutputOwnership:
    """Coordinate output starts and exact-owner retirement on the app's event loop."""

    def __init__(self) -> None:
        self._slot: tuple[AudioOutputLease, _Retire] | None = None
        self._generation = 0
        self._blocked = False
        self._retirement: asyncio.Task[bool] | None = None
        self._shutdown: asyncio.Future[bool] | None = None
        self._handoff_epoch: object | None = None
        self._intent: AudioOutputIntent | None = None
        self._serial = 0
        self._eligible_serial = 0
        self._tentative: WeakSet[AudioOutputIntent] = WeakSet()

    @property
    def closing(self) -> bool:
        """Return whether shutdown has fenced all future output."""
        return self._shutdown is not None

    @property
    def handoff_epoch(self) -> object | None:
        """Return the opaque identity of the latest unresolved external playback."""
        return self._handoff_epoch

    def begin_intent(self, *, tentative: bool = False) -> AudioOutputIntent:
        """Reserve action order before async preparation, without stopping audio."""
        if self.closing:
            raise OutputClosed("Audio output admission has closed")
        self._serial += 1
        ticket = AudioOutputIntent(self._serial, self)
        if tentative:
            self._tentative.add(ticket)
        else:
            self._intent = ticket
            self._eligible_serial = ticket.serial
        return ticket

    def acknowledge_handoff(self, ticket: AudioOutputIntent, epoch: object) -> bool:
        """Accept a positive user response for the exact pending action and epoch."""
        if (
            self.closing
            or self._intent is None
            or self._intent is not ticket
            or self._handoff_epoch is None
            or self._handoff_epoch is not epoch
        ):
            return False
        self._handoff_epoch = None
        return True

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
    ) -> bool:
        # An eager task can enter here before create_task returns to the host.
        await published.wait()
        try:
            confirmed = await retire()
        except (Exception, asyncio.CancelledError):
            confirmed = False
        if confirmed is True:
            lease.release()
        elif self._owns(lease):
            self._blocked = True
        return not self._owns(lease)

    def _start_retirement(
        self, slot: tuple[AudioOutputLease, _Retire]
    ) -> asyncio.Task[bool]:
        slot[0].revoke()
        if self._retirement is None or self._retirement.done():
            published = asyncio.Event()
            self._retirement = asyncio.create_task(self._retire_owned(*slot, published))
            published.set()
        return self._retirement

    @staticmethod
    def _resolved(confirmed: bool) -> asyncio.Future[bool]:
        result = asyncio.get_running_loop().create_future()
        result.set_result(confirmed)
        return result

    def stop(self, lease: AudioOutputLease) -> asyncio.Future[bool]:
        """Fence the exact held output now and observe its checked retirement."""
        if self._slot is None or self._slot[0] is not lease:
            return self._resolved(False)
        return asyncio.shield(self._start_retirement(self._slot))

    def begin_shutdown(self) -> asyncio.Future[bool]:
        """Fence admission now and share one cancellation-safe retirement result."""
        if self._shutdown is None:
            self._intent = None
            self._tentative.clear()
            if self._slot is not None:
                self._shutdown = self._start_retirement(self._slot)
            elif self._retirement is not None and not self._retirement.done():
                self._shutdown = self._retirement
            else:
                self._shutdown = self._resolved(True)
        return asyncio.shield(self._shutdown)

    def _check_request(
        self, ticket: AudioOutputIntent, generation: int | None = None
    ) -> None:
        if self.closing:
            raise OutputClosed("Audio output admission has closed")
        if self._intent is not ticket or (
            generation is not None and self._generation != generation
        ):
            raise OutputSuperseded("Audio output request was superseded")

    def _check_output(self, external: bool) -> None:
        if self._blocked and self._slot is not None:
            raise OutputBlocked(self._slot[0]._owner)
        if not external and self._handoff_epoch is not None:
            raise OutputHandoffRequired(self._handoff_epoch)

    async def acquire(
        self,
        owner: str,
        retire: _Retire,
        *,
        intent: AudioOutputIntent | None = None,
        external: bool = False,
    ) -> AudioOutputLease:
        """Revoke earlier starts and wait for the previous output to retire.

        Args:
            owner: Diagnostic label for this start.
            retire: Callback that retires this start's exact resource and returns
                True after confirming it stopped. Move blocking work off-loop.
            intent: Action reserved before preparation. If omitted, reserve one now.
            external: This action explicitly delegates playback to the OS.

        Returns:
            The lease authorizing the newest start on the app's event loop.

        Raises:
            OutputSuperseded: The ticket or this acquisition has been superseded.
            OutputClosed: The app began shutdown.
            OutputBlocked: An owner's retirement is unconfirmed.
            OutputHandoffRequired: External playback needs user acknowledgement.
            asyncio.CancelledError: The caller cancelled its request.
        """
        ticket = self.begin_intent() if intent is None else intent
        self._check_request(ticket)
        self._generation += 1
        generation = self._generation
        self._check_output(external)
        if self._slot is not None:
            self._start_retirement(self._slot)
        while (observed := self._retirement) is not None:
            # An explicit Stop may have published a retry before this waiter resumes.
            await asyncio.shield(observed)
            self._check_request(ticket, generation)
            if self._retirement is observed:
                self._retirement = None
        self._check_output(external)
        lease = AudioOutputLease(generation, self, owner)
        self._slot = lease, retire
        return lease
