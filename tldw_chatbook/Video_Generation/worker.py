"""Request builder + blocking generation entry. Callers (the Console command,
and later the demo screen) must call run_generation() from a thread worker —
never on the UI loop, because the adapters are synchronous and blocking.
"""
from __future__ import annotations

from typing import Any

from tldw_chatbook.Video_Generation.adapter_registry import get_registry
from tldw_chatbook.Video_Generation.adapters.base import (
    ResolvedReferenceAsset,
    VideoGenRequest,
    VideoGenResult,
)
from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError
from tldw_chatbook.Video_Generation.request_validation import validate_video_generation_request


def build_request(
    *,
    backend: str,
    prompt: str,
    negative_prompt: str | None = None,
    duration_seconds: int | None = None,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    ratio: str | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    sampler: str | None = None,
    model: str | None = None,
    video_format: str = "mp4",
    extra_params: dict[str, Any] | None = None,
    reference_assets: tuple[ResolvedReferenceAsset, ...] = (),
) -> VideoGenRequest:
    """Build a :class:`VideoGenRequest` from caller/UI inputs.

    Args:
        backend: Backend name (must be enabled in config).
        prompt: Positive prompt text.
        negative_prompt: Optional negative prompt.
        duration_seconds: Optional clip length in seconds.
        fps: Optional frames per second.
        width: Optional video width in pixels.
        height: Optional video height in pixels.
        ratio: Optional aspect ratio (``"16:9"`` …) or ``"adaptive"``.
        steps: Optional sampling steps (local backends).
        cfg_scale: Optional classifier-free-guidance scale (local backends).
        seed: Optional seed (``-1`` for random).
        sampler: Optional sampler name (local backends).
        model: Optional model override.
        video_format: Output container format (defaults to ``"mp4"``).
        extra_params: Backend-specific passthrough params (coerced to ``{}`` if None).
        reference_assets: Optional resolved reference assets (first/last
            frame, reference image/video/audio). Validated at the
            ``run_generation`` choke point before any adapter sees them.

    Returns:
        A frozen :class:`VideoGenRequest`.
    """
    return VideoGenRequest(
        backend=backend, prompt=prompt, negative_prompt=negative_prompt,
        duration_seconds=duration_seconds, fps=fps, width=width, height=height,
        ratio=ratio, steps=steps, cfg_scale=cfg_scale, seed=seed,
        sampler=sampler, model=model, format=video_format,
        extra_params=dict(extra_params or {}),
        reference_assets=tuple(reference_assets),
    )


def run_generation(request: VideoGenRequest) -> VideoGenResult:
    """Validate, resolve the backend, and invoke its adapter. Blocking.

    Enforces the request-validation layer (bounds + per-backend
    ``extra_params`` allowlist, plus reference-asset kind/mime/size/count
    checks when ``request.reference_assets`` is non-empty) at this single
    entry point *before* dispatch, so a caller that constructs a
    :class:`VideoGenRequest` directly cannot bypass it.

    Must run on a thread — the adapters are synchronous and blocking.

    Args:
        request: The video generation request.

    Returns:
        The generated :class:`VideoGenResult`.

    Raises:
        VideoGenerationError: If the backend is not enabled/available, the
            request fails validation, or the adapter fails to load.
    """
    registry = get_registry()
    resolved = registry.resolve_backend(request.backend)
    if resolved is None:
        raise VideoGenerationError(
            f"Backend {request.backend!r} is not enabled/available. "
            f"Check [video_generation].enabled_backends."
        )
    issues = validate_video_generation_request(
        {
            "backend": resolved,
            "prompt": request.prompt,
            "duration_seconds": request.duration_seconds,
            "fps": request.fps,
            "width": request.width,
            "height": request.height,
            "ratio": request.ratio,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "extra_params": request.extra_params,
            "reference_assets": request.reference_assets,
        }
    )
    if issues:
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise VideoGenerationError(f"Invalid video generation request: {detail}")
    adapter = registry.get_adapter(resolved)
    if adapter is None:
        raise VideoGenerationError(f"Adapter for backend {resolved!r} failed to load.")
    return adapter.generate(request)
