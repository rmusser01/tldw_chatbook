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

from ...Model_Artifacts.service import ArtifactRef


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


class AudioCppModelInstallOwner:
    """Retain audio.cpp installation work through screen and app teardown."""

    def __init__(self) -> None:
        self._active: set[AudioCppModelInstallOperation] = set()
        self._sealed = False

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

        while self._active:
            await asyncio.shield(
                asyncio.gather(
                    *(operation.task for operation in tuple(self._active)),
                    return_exceptions=True,
                )
            )

    async def shutdown(self) -> None:
        """Seal the owner, request cancellation, and drain every operation."""

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
