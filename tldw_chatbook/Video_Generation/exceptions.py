"""Exceptions for video generation adapters."""


class VideoGenerationError(RuntimeError):
    """Raised when video generation fails."""

    def __init__(self, message: str = "video generation failed") -> None:
        super().__init__(message)


class VideoBackendUnavailableError(VideoGenerationError):
    """Raised when a video backend is not configured or available."""

    def __init__(self, message: str = "video backend unavailable") -> None:
        super().__init__(message)
