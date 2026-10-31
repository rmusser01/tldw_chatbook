"""Exceptions for video generation adapters."""

from typing import Literal

RemoteCancelOutcome = Literal["cancelled", "refused", "unconfirmed"]
"""What a paid remote backend did with a Stop (TASK-33621.50).

``cancelled``: the vendor confirmed it cancelled the task, so it is not
billed. ``refused``: the vendor refused (MiniMax does once a task has started
rendering), so the task keeps running and may still be billed. ``unconfirmed``:
the cancel request failed or its answer was not a cancel, so the task may
still run and be billed.
"""


class VideoGenerationError(RuntimeError):
    """Raised when video generation fails."""

    def __init__(self, message: str = "video generation failed") -> None:
        super().__init__(message)


class VideoBackendUnavailableError(VideoGenerationError):
    """Raised when a video backend is not configured or available."""

    def __init__(self, message: str = "video backend unavailable") -> None:
        super().__init__(message)


class VideoGenerationCancelled(VideoGenerationError):
    """Raised when a generation stops because the caller cancelled it.

    A stop, not a failure (TASK-33621.45): callers report it as stopped.

    Attributes:
        remote_cancel: For a paid remote backend, what the vendor did with
            the cancel (see ``RemoteCancelOutcome``); ``None`` for a local
            backend, where stopping costs nothing.
    """

    def __init__(
        self,
        message: str = "video generation cancelled by user",
        *,
        remote_cancel: RemoteCancelOutcome | None = None,
    ) -> None:
        super().__init__(message)
        self.remote_cancel = remote_cancel
