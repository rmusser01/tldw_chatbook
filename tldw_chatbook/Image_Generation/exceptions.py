"""Exceptions for image generation adapters."""

class ImageGenerationError(RuntimeError):
    """Raised when image generation fails."""

    def __init__(self, message: str = "image generation failed") -> None:
        super().__init__(message)


class ImageBackendUnavailableError(ImageGenerationError):
    """Raised when an image backend is not configured or available."""

    def __init__(self, message: str = "image backend unavailable") -> None:
        super().__init__(message)
