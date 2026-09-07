"""Exceptions for image generation adapters."""

from typing import Literal


ComfyUIImageEditPhase = Literal[
    "request_validation",
    "packaged_workflow_validation",
    "remote_schema_preflight",
    "source_upload",
    "prompt_submission",
    "history_polling",
    "output_descriptor_validation",
    "output_download",
    "persistence",
]

_COMFYUI_IMAGE_EDIT_GUIDANCE: dict[ComfyUIImageEditPhase, str] = {
    "request_validation": "The image-edit request is invalid. Check the source image and controls.",
    "packaged_workflow_validation": "The packaged image-edit workflow is unavailable.",
    "remote_schema_preflight": "The image-edit server is not compatible with this workflow.",
    "source_upload": "The source image could not be uploaded. Please try again.",
    "prompt_submission": "The image-edit request could not be submitted. Please try again.",
    "history_polling": "The image-edit operation did not complete. Please try again.",
    "output_descriptor_validation": "The image-edit server returned an invalid output.",
    "output_download": "The edited image could not be downloaded safely.",
    "persistence": "The edited image could not be saved locally. The source remains staged.",
}

class ImageGenerationError(RuntimeError):
    """Raised when image generation fails."""

    def __init__(self, message: str = "image generation failed") -> None:
        super().__init__(message)


class ImageBackendUnavailableError(ImageGenerationError):
    """Raised when an image backend is not configured or available."""

    def __init__(self, message: str = "image backend unavailable") -> None:
        super().__init__(message)


class ImageGenerationCancelled(ImageGenerationError):
    """Raised when image generation observes caller cancellation."""

    def __init__(self, message: str = "image generation cancelled") -> None:
        super().__init__(message)


class ComfyUIImageEditError(ImageGenerationError):
    """Privacy-safe image-edit failure carrying a closed operation phase."""

    def __init__(self, phase: ComfyUIImageEditPhase) -> None:
        guidance = _COMFYUI_IMAGE_EDIT_GUIDANCE.get(phase)
        if guidance is None:
            raise ValueError("unknown image-edit failure phase")
        self.phase = phase
        super().__init__(guidance)
