"""Request builder + blocking generation entry. Callers (the Console command,
and later the demo screen) must call run_generation() from a thread worker —
never on the UI loop, because the adapters are synchronous and blocking.
"""
from __future__ import annotations

import inspect
import threading
from typing import Any

from tldw_chatbook.Video_Generation.adapter_registry import get_registry
from tldw_chatbook.Video_Generation.adapters.base import (
    ResolvedReferenceAsset,
    VideoGenRequest,
    VideoGenResult,
)
from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError
from tldw_chatbook.Video_Generation.request_validation import validate_video_generation_request
from tldw_chatbook.Video_Generation.video_formats import (
    canonical_video_extension,
    video_container_for_mime,
)


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


def _generate_with_optional_cancel(
    adapter: Any, request: VideoGenRequest, cancel_event: threading.Event | None
) -> VideoGenResult:
    """Dispatch to the adapter, threading ``cancel_event`` only when supported.

    Adapters with cooperative cancellation (minimax, task-3401.3) declare a
    ``cancel_event`` keyword; adapters that predate it are called without
    it. Detected by signature (never by catching TypeError from inside
    ``generate``, which would misclassify an adapter's own type errors).
    """
    if cancel_event is not None:
        try:
            parameters = inspect.signature(adapter.generate).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts = "cancel_event" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts:
            return adapter.generate(request, cancel_event=cancel_event)
    return adapter.generate(request)


def run_generation(
    request: VideoGenRequest,
    *,
    cancel_event: threading.Event | None = None,
) -> VideoGenResult:
    """Validate, resolve the backend, and invoke its adapter. Blocking.

    Enforces the request-validation layer (bounds + per-backend
    ``extra_params`` allowlist, plus reference-asset kind/mime/size/count
    checks when ``request.reference_assets`` is non-empty) at this single
    entry point *before* dispatch, so a caller that constructs a
    :class:`VideoGenRequest` directly cannot bypass it.

    Must run on a thread — the adapters are synchronous and blocking.

    Args:
        request: The video generation request.
        cancel_event: Optional cooperative-cancellation event, threaded to
            adapters that declare support for it (minimax); silently not
            passed to adapters that don't.

    Returns:
        The generated :class:`VideoGenResult`.

    Raises:
        VideoGenerationError: If the backend is not enabled/available, the
            request fails validation, or the adapter fails to load.
    """
    try:
        canonical_video_extension(request.format)
    except ValueError as exc:
        raise VideoGenerationError("Invalid video generation request format") from exc

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
    result = _generate_with_optional_cancel(adapter, request, cancel_event)
    try:
        result_container = result.container
        canonical_video_extension(result_container)
        mime_container = video_container_for_mime(result.content_type)
    except Exception:
        raise VideoGenerationError("Invalid video generation result format") from None
    if request.format != result_container or result_container != mime_container:
        raise VideoGenerationError("Invalid video generation result format")
    return result
