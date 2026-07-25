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
    #: The seed the backend actually used, when (and only when) the adapter
    #: can determine it without guessing -- e.g. reading it back out of the
    #: backend's response. ``None`` (the default) covers both "the request's
    #: own seed was used verbatim" and "the backend doesn't expose this";
    #: callers fall back to the request's seed either way (task-558). Every
    #: existing adapter leaves this ``None`` today -- none of the six
    #: backends' response parsing reports a resolved seed reliably enough to
    #: surface without risking a wrong value on the card.
    resolved_seed: int | None = None
    #: The model the adapter actually used for this request, when the
    #: adapter can state that with certainty (e.g. it already resolved
    #: ``request.model`` against a configured default/local file before
    #: sending) -- never a value merely echoed by a remote response unless
    #: verified. ``None`` when the adapter doesn't populate this (task-558).
    resolved_model: str | None = None


class ImageGenerationAdapter(Protocol):
    """Protocol for image generation backends."""

    name: str
    supported_formats: set[str]

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        """Generate an image from the given request."""
        ...
