"""Request builder + blocking generation entry. The Textual demo screen (and, in
Phase 2, the chat card) call run_generation() from a thread worker — never on the
UI loop, because the adapters are synchronous and blocking.
"""
from __future__ import annotations
from tldw_chatbook.Image_Generation.adapter_registry import get_registry
from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError


def build_request(*, backend, prompt, negative_prompt=None, width=None, height=None,
                  steps=None, cfg_scale=None, seed=None, sampler=None, model=None,
                  image_format="png", extra_params=None) -> ImageGenRequest:
    return ImageGenRequest(
        backend=backend, prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, steps=steps, cfg_scale=cfg_scale, seed=seed,
        sampler=sampler, model=model, format=image_format,
        extra_params=dict(extra_params or {}),
    )


def run_generation(request: ImageGenRequest) -> ImageGenResult:
    """Blocking. Resolve the backend and invoke its adapter. Raises ImageGenerationError."""
    registry = get_registry()
    resolved = registry.resolve_backend(request.backend)
    if resolved is None:
        raise ImageGenerationError(
            f"Backend {request.backend!r} is not enabled/available. "
            f"Check [image_generation].enabled_backends."
        )
    adapter = registry.get_adapter(resolved)
    if adapter is None:
        raise ImageGenerationError(f"Adapter for backend {resolved!r} failed to load.")
    return adapter.generate(request)
