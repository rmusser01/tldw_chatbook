"""Base contracts for video generation adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

#: Role a reference asset plays in a video generation request. Mirrors the
#: MiniMax-H3 ``content[]`` roles; local backends (sd.cpp, ComfyUI) use the
#: subset their models support (typically ``first_frame`` for image-to-video).
ReferenceAssetKind = Literal[
    "first_frame",
    "last_frame",
    "reference_image",
    "reference_video",
    "reference_audio",
]


@dataclass(frozen=True)
class ResolvedReferenceAsset:
    """One fully-resolved reference asset for a video request.

    Adapters can assume any asset that reaches them has already passed the
    choke-point checks (kind count/mime/size/non-empty content) in
    ``request_validation.py`` -- the same guarantee the image package gives
    for ``ResolvedReferenceImage``.

    Attributes:
        kind: The role this asset plays in generation.
        content: Raw asset bytes (never empty).
        mime_type: MIME type of ``content`` (e.g. ``image/png``, ``video/mp4``).
        source_name: Human-readable origin label (e.g. the kept image
            variant's display name) for cards and logs -- never a path.
    """

    kind: ReferenceAssetKind
    content: bytes
    mime_type: str
    source_name: str = ""


@dataclass(frozen=True)
class VideoGenRequest:
    backend: str
    prompt: str
    negative_prompt: str | None
    duration_seconds: int | None
    fps: int | None
    width: int | None
    height: int | None
    ratio: str | None
    steps: int | None
    cfg_scale: float | None
    seed: int | None
    sampler: str | None
    model: str | None
    format: str
    extra_params: dict[str, Any]
    request_id: str | None = None
    reference_assets: tuple[ResolvedReferenceAsset, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VideoGenResult:
    content: bytes
    content_type: str
    container: str
    bytes_len: int
    duration_seconds: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    #: The seed the backend actually used, when (and only when) the adapter
    #: can determine it without guessing. ``None`` covers both "the request's
    #: own seed was used verbatim" and "the backend doesn't expose this";
    #: callers fall back to the request's seed either way (same rule as the
    #: image package's task-558 contract).
    resolved_seed: int | None = None
    #: The model the adapter actually used, when the adapter can state that
    #: with certainty -- never a value merely echoed by a remote response
    #: unless verified. ``None`` when the adapter doesn't populate this.
    resolved_model: str | None = None


class VideoGenerationAdapter(Protocol):
    """Protocol for video generation backends."""

    name: str
    supported_formats: set[str]

    def generate(self, request: VideoGenRequest) -> VideoGenResult:
        """Generate a video from the given request."""
        ...
