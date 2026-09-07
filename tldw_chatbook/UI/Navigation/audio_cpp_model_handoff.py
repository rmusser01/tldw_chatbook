"""Process-local typed handoff values for audio.cpp Model Library selection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from pathlib import PurePosixPath, PureWindowsPath
import re
import threading
from typing import Awaitable, Callable
import unicodedata

from ...Model_Artifacts.service import ArtifactRef, LeasedArtifactHandle


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z", re.ASCII)
_LOGGER = logging.getLogger(__name__)


def _validate_token(value: object) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError("audio.cpp Model Library token is invalid")


def _validate_draft_revision(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("audio.cpp Model Library draft revision is invalid")


def _validate_canonical_root(value: object) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 4096
        or any(
            character in {"\x00", "\r", "\n"}
            or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            for character in value
        )
    ):
        raise ValueError("audio.cpp Model Library root is invalid")
    windows_spelling = value.replace("/", "\\")
    if windows_spelling.startswith(("\\\\?\\", "\\\\.\\")):
        raise ValueError("audio.cpp Model Library root is invalid")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if not (posix.is_absolute() or windows.is_absolute()):
        raise ValueError("audio.cpp Model Library root must be absolute")
    if any(part in {".", ".."} for part in (*posix.parts, *windows.parts)):
        raise ValueError("audio.cpp Model Library root must be canonical")
    if posix.is_absolute() and (
        len(posix.parts) < 2 or "\\" in value or posix.as_posix() != value
    ):
        raise ValueError("audio.cpp Model Library root must be canonical")
    if windows.is_absolute() and (len(windows.parts) < 2 or str(windows) != value):
        raise ValueError("audio.cpp Model Library root must be canonical")


@dataclass(frozen=True, slots=True)
class AudioCppModelLibraryRequest:
    """Opaque Settings request to browse reviewed audio.cpp packages."""

    token: str
    draft_revision: int

    def __post_init__(self) -> None:
        _validate_token(self.token)
        _validate_draft_revision(self.draft_revision)


@dataclass(frozen=True, slots=True)
class AudioCppModelLibraryResult:
    """Exact installed package returned to the originating Settings draft."""

    token: str
    draft_revision: int
    artifact_id: str
    revision: str
    variant: str
    canonical_root: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_token(self.token)
        _validate_draft_revision(self.draft_revision)
        ArtifactRef(self.artifact_id, self.revision, self.variant)
        _validate_canonical_root(self.canonical_root)


AudioCppInstallRunner = Callable[
    [threading.Event], Awaitable[AudioCppModelLibraryResult | None]
]
AudioCppInstallSettled = Callable[
    [AudioCppModelLibraryResult | None, BaseException | None, bool], None
]


@dataclass(eq=False, slots=True)
class AudioCppModelInstallOperation:
    """One app-owned audio.cpp install and its cooperative cancellation."""

    cancel_event: threading.Event
    task: asyncio.Task[None] = field(init=False, repr=False)


@dataclass(eq=False, slots=True)
class AudioCppManagedLeaseHold:
    """One exact app-owned Settings lease hold and its cleanup state."""

    handles: list[LeasedArtifactHandle] = field(default_factory=list, repr=False)
    acquisition_task: asyncio.Task[None] = field(init=False, repr=False)
    cleanup_task: asyncio.Task[None] | None = field(default=None, repr=False)
    release_requested: bool = False
    acquisition_error: BaseException | None = field(default=None, repr=False)
    cleanup_control_error: BaseException | None = field(default=None, repr=False)
    publication_adopted: bool = False


@dataclass(eq=False, slots=True)
class AudioCppManagedLeasePublication:
    """Process-local transfer of one exact hold into TTS publication."""

    owner: AudioCppModelInstallOwner
    hold: AudioCppManagedLeaseHold

    def adopt(self) -> None:
        """Mark the service as the hold's definitive lifecycle owner."""

        if self.hold not in self.owner._lease_holds:
            raise RuntimeError("audio.cpp model publication lease is unavailable")
        self.hold.publication_adopted = True

    async def release(self) -> None:
        """Request and join one exact cleanup attempt.

        Raises:
            RuntimeError: If exact handle cleanup remains incomplete.
            BaseException: If cleanup reports interpreter control flow.
        """

        self.owner.request_lease_release(self.hold)
        await self.owner.wait_lease_hold(self.hold)
        if self.hold in self.owner._lease_holds:
            raise RuntimeError("audio.cpp model publication cleanup failed")

    def abandon(self) -> None:
        """Return an unadopted transfer to app-owned cleanup."""

        if self.hold.publication_adopted:
            return
        self.owner.request_lease_release(self.hold)


class AudioCppModelInstallOwner:
    """Retain audio.cpp installation and lease work through app teardown."""

    def __init__(self) -> None:
        self._active: set[AudioCppModelInstallOperation] = set()
        self._lease_holds: set[AudioCppManagedLeaseHold] = set()
        self._sealed = False

    @property
    def cleanup_pending(self) -> bool:
        """Return whether exact managed lease cleanup remains owned."""

        return bool(self._lease_holds)

    @staticmethod
    def _close_cleanup_handles(
        handles: tuple[LeasedArtifactHandle, ...],
    ) -> tuple[list[LeasedArtifactHandle], BaseException | None]:
        remaining: list[LeasedArtifactHandle] = []
        first_error: BaseException | None = None
        for handle in handles:
            try:
                handle.close()
            except BaseException as error:
                remaining.append(handle)
                if first_error is None or (
                    isinstance(first_error, Exception)
                    and not isinstance(error, Exception)
                ):
                    first_error = error
        return remaining, first_error

    async def acquire_lease_hold(
        self,
        references: tuple[ArtifactRef, ...],
        service_factory: Callable[[], object],
    ) -> AudioCppManagedLeaseHold:
        """Acquire exact inactive roots under immediate app ownership.

        Args:
            references: Exact managed artifact identities to lease.
            service_factory: Factory for the app's model-artifact service.

        Returns:
            The registered hold controlling release of the acquired handles.

        Raises:
            asyncio.CancelledError: If the waiting panel operation is cancelled.
            RuntimeError: If acquisition fails or the owner is shut down.
            BaseException: If acquisition reports interpreter control flow.
        """

        if self._sealed:
            raise RuntimeError("audio.cpp model cleanup owner is shut down")
        hold = AudioCppManagedLeaseHold()
        self._lease_holds.add(hold)

        async def acquire() -> None:
            try:
                service = await asyncio.to_thread(service_factory)
                acquire_root = getattr(service, "acquire_installed_root")
                for reference in references:
                    handle = await asyncio.to_thread(acquire_root, reference)
                    hold.handles.append(handle)
            except BaseException as error:
                hold.acquisition_error = error
            finally:
                if hold.release_requested or hold.acquisition_error is not None:
                    self._start_hold_cleanup(hold)

        hold.acquisition_task = asyncio.create_task(
            acquire(),
            name="audio-cpp-model-lease-acquire",
        )
        try:
            await asyncio.shield(hold.acquisition_task)
        except asyncio.CancelledError:
            self.request_lease_release(hold)
            try:
                await asyncio.shield(hold.acquisition_task)
                await self.wait_lease_hold(hold)
            except BaseException:
                # The original cancellation remains authoritative. Any failed
                # cleanup stays registered for the app lifecycle to retry.
                pass
            raise
        if hold.acquisition_error is not None:
            self.request_lease_release(hold)
            await self.wait_lease_hold(hold)
            error = hold.acquisition_error
            if error is not None and not isinstance(error, Exception):
                raise error
            raise RuntimeError("audio.cpp model lease acquisition failed") from None
        return hold

    def request_lease_release(self, hold: AudioCppManagedLeaseHold) -> None:
        """Request cleanup of one exact registered hold idempotently.

        Args:
            hold: The exact operation returned by :meth:`acquire_lease_hold`.
        """

        if type(hold) is not AudioCppManagedLeaseHold or hold not in self._lease_holds:
            return
        hold.release_requested = True
        if hold.acquisition_task.done():
            self._start_hold_cleanup(hold)

    def transfer_lease_hold_to_publication(
        self,
        hold: AudioCppManagedLeaseHold,
    ) -> AudioCppManagedLeasePublication:
        """Create an opaque process-local publication transfer.

        Args:
            hold: The exact acquired Settings hold to transfer.

        Returns:
            A single-use service adoption and release operation.

        Raises:
            RuntimeError: If the hold is not actively owned.
        """

        if hold not in self._lease_holds or hold.release_requested:
            raise RuntimeError("audio.cpp model lease hold is unavailable")
        return AudioCppManagedLeasePublication(self, hold)

    def retry_cleanup(self) -> None:
        """Start one bounded retry when retained cleanup is idle."""

        for hold in tuple(self._lease_holds):
            if hold.release_requested and hold.acquisition_task.done():
                self._start_hold_cleanup(hold)

    def _start_hold_cleanup(self, hold: AudioCppManagedLeaseHold) -> None:
        if hold.cleanup_task is not None and not hold.cleanup_task.done():
            return
        if not hold.handles:
            self._lease_holds.discard(hold)
            return
        claimed = tuple(hold.handles)
        hold.cleanup_control_error = None

        async def owned_cleanup() -> None:
            remaining, error = await asyncio.to_thread(
                self._close_cleanup_handles,
                claimed,
            )
            hold.handles = remaining
            hold.cleanup_task = None
            if error is not None and not isinstance(error, Exception):
                hold.cleanup_control_error = error
            if not remaining:
                self._lease_holds.discard(hold)

        hold.cleanup_task = asyncio.create_task(
            owned_cleanup(),
            name="audio-cpp-model-lease-cleanup",
        )

    async def wait_lease_hold(self, hold: AudioCppManagedLeaseHold) -> None:
        """Wait for the hold's current acquisition and cleanup attempts.

        Args:
            hold: The exact registered hold to observe.

        Raises:
            BaseException: If handle cleanup reports interpreter control flow.
        """

        await asyncio.shield(hold.acquisition_task)
        task = hold.cleanup_task
        if task is not None:
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        if hold.cleanup_control_error is not None:
            raise hold.cleanup_control_error

    @property
    def active_count(self) -> int:
        """Return the number of actual operations not yet settled."""

        return len(self._active)

    def start(
        self,
        runner: AudioCppInstallRunner,
        on_settled: AudioCppInstallSettled,
    ) -> AudioCppModelInstallOperation:
        """Start and retain one operation until its runner truly settles."""

        if self._sealed:
            raise RuntimeError("audio.cpp install owner is shut down")
        if not callable(runner) or not callable(on_settled):
            raise TypeError("audio.cpp install runner and callback must be callable")
        operation = AudioCppModelInstallOperation(threading.Event())

        async def owned() -> None:
            result: AudioCppModelLibraryResult | None = None
            error: BaseException | None = None
            try:
                result = await runner(operation.cancel_event)
            except BaseException as exc:
                error = exc
            cancelled = operation.cancel_event.is_set() or isinstance(
                error, asyncio.CancelledError
            )
            if cancelled:
                result = None
                if isinstance(error, asyncio.CancelledError):
                    error = None
            try:
                on_settled(result, error, cancelled)
            except Exception as exc:  # pragma: no cover - containment seam
                _LOGGER.error(
                    "audio.cpp install settlement callback failed; error_type=%s",
                    type(exc).__name__,
                )
            finally:
                self._active.discard(operation)

        operation.task = asyncio.create_task(owned(), name="audio-cpp-model-install")
        self._active.add(operation)
        return operation

    def request_cancel(self, operation: AudioCppModelInstallOperation) -> None:
        """Request cooperative cancellation without detaching the runner."""

        if type(operation) is not AudioCppModelInstallOperation:
            raise TypeError("invalid audio.cpp install operation")
        operation.cancel_event.set()

    async def wait(self, operation: AudioCppModelInstallOperation) -> None:
        """Wait for definitive runner and settlement-callback completion."""

        if type(operation) is not AudioCppModelInstallOperation:
            raise TypeError("invalid audio.cpp install operation")
        await asyncio.shield(operation.task)

    async def wait_until_idle(self) -> None:
        """Wait until every operation active at each observation has settled."""

        while self._active or any(
            not hold.acquisition_task.done() or hold.cleanup_task is not None
            for hold in self._lease_holds
        ):
            tasks = [operation.task for operation in tuple(self._active)]
            for hold in self._lease_holds:
                if not hold.acquisition_task.done():
                    tasks.append(hold.acquisition_task)
                if hold.cleanup_task is not None:
                    tasks.append(hold.cleanup_task)
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
            for hold in self._lease_holds:
                if hold.cleanup_control_error is not None:
                    raise hold.cleanup_control_error

    async def shutdown(self) -> None:
        """Seal the owner, request cancellation, and drain every operation.

        Raises:
            RuntimeError: If exact managed handles remain after bounded retries.
            BaseException: If cleanup reports interpreter control flow.
        """

        self._sealed = True
        operations = tuple(self._active)
        for operation in operations:
            operation.cancel_event.set()
        if operations:
            await asyncio.shield(
                asyncio.gather(
                    *(operation.task for operation in operations),
                    return_exceptions=True,
                )
            )
        owned_holds = tuple(
            hold
            for hold in self._lease_holds
            if not hold.publication_adopted or hold.release_requested
        )
        for hold in owned_holds:
            self.request_lease_release(hold)
        if owned_holds:
            self.retry_cleanup()
            await self.wait_until_idle()
        if any(hold in self._lease_holds for hold in owned_holds):
            raise RuntimeError("audio.cpp model cleanup failed")
