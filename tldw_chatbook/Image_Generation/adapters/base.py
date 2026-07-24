"""Base contracts for image generation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage


@dataclass(frozen=True)
class ImageGenRequest:
    backend: str
    prompt: str
    negative_prompt: str | None
    width: int | None
    height: int | None
    steps: int | None
    cfg_scale: float | None
    seed: int | None
    sampler: str | None
    model: str | None
    format: str
    extra_params: dict[str, Any]
    request_id: str | None = None
    reference_image: ResolvedReferenceImage | None = None


@dataclass(frozen=True)
class ImageGenResult:
    content: bytes
    content_type: str
    bytes_len: int


class ImageGenerationAdapter(Protocol):
    """Protocol for image generation backends."""

    name: str
    supported_formats: set[str]

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        """Generate an image from the given request."""
        ...
