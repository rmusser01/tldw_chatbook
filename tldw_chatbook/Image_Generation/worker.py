"""Request builder + blocking generation entry. The Textual demo screen (and, in
Phase 2, the chat card) call run_generation() from a thread worker — never on the
UI loop, because the adapters are synchronous and blocking.
"""
from __future__ import annotations
from typing import Any
from tldw_chatbook.Image_Generation.adapter_registry import get_registry
from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
from tldw_chatbook.Image_Generation.request_validation import validate_image_generation_request


def build_request(
    *,
    backend: str,
    prompt: str,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    sampler: str | None = None,
    model: str | None = None,
    image_format: str = "png",
    extra_params: dict[str, Any] | None = None,
) -> ImageGenRequest:
    """Build an :class:`ImageGenRequest` from caller/UI inputs.

    Args:
        backend: Backend name (must be enabled in config).
        prompt: Positive prompt text.
        negative_prompt: Optional negative prompt.
        width: Optional image width in pixels.
        height: Optional image height in pixels.
        steps: Optional sampling steps.
        cfg_scale: Optional classifier-free-guidance scale.
        seed: Optional seed (``-1`` for random).
        sampler: Optional sampler name.
        model: Optional model override.
        image_format: Output format (defaults to ``"png"``).
        extra_params: Backend-specific passthrough params (coerced to ``{}`` if None).

    Returns:
        A frozen :class:`ImageGenRequest`.
    """
    return ImageGenRequest(
        backend=backend, prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, steps=steps, cfg_scale=cfg_scale, seed=seed,
        sampler=sampler, model=model, format=image_format,
        extra_params=dict(extra_params or {}),
    )


def run_generation(request: ImageGenRequest) -> ImageGenResult:
    """Validate, resolve the backend, and invoke its adapter. Blocking.

    Enforces the request-validation layer (bounds + per-backend ``extra_params``
    allowlist) at this single entry point *before* dispatch, so a caller that
    constructs an :class:`ImageGenRequest` directly cannot bypass it (e.g. the
    stable-diffusion.cpp ``cli_args`` passthrough).

    Must run on a thread — the adapters are synchronous and blocking.

    Args:
        request: The image generation request.

    Returns:
        The generated :class:`ImageGenResult`.

    Raises:
        ImageGenerationError: If the backend is not enabled/available, the
            request fails validation, or the adapter fails to load.
    """
    registry = get_registry()
    resolved = registry.resolve_backend(request.backend)
    if resolved is None:
        raise ImageGenerationError(
            f"Backend {request.backend!r} is not enabled/available. "
            f"Check [image_generation].enabled_backends."
        )
    issues = validate_image_generation_request(
        {
            "backend": resolved,
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "extra_params": request.extra_params,
        }
    )
    if issues:
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise ImageGenerationError(f"Invalid image generation request: {detail}")
    adapter = registry.get_adapter(resolved)
    if adapter is None:
        raise ImageGenerationError(f"Adapter for backend {resolved!r} failed to load.")
    return adapter.generate(request)
