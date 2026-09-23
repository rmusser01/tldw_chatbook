"""Registry for image generation backends (skeleton in Media_Generation)."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

from tldw_chatbook.Image_Generation.adapters.base import ImageGenerationAdapter
from tldw_chatbook.Image_Generation.config import (
    _IMAGE_GENERATION_RUNTIME_LOCK,
    _use_image_generation_config_snapshot,
    get_image_generation_config,
)
from tldw_chatbook.Media_Generation.adapter_registry import MediaAdapterRegistry


class ImageAdapterRegistry(MediaAdapterRegistry):
    """Registry for image generation adapters.

    ADR-176: the shared skeleton (lazy resolution, enablement, per-backend
    caching, register/reset lifecycle) lives in
    ``Media_Generation.adapter_registry``; this class contributes the image
    backend table and the config-snapshot wiring.
    """

    modality = "image"
    log_error_detail = True

    DEFAULT_ADAPTERS: dict[str, str] = {
        "stable_diffusion_cpp": "tldw_chatbook.Image_Generation.adapters.stable_diffusion_cpp_adapter.StableDiffusionCppAdapter",
        "swarmui": "tldw_chatbook.Image_Generation.adapters.swarmui_adapter.SwarmUIAdapter",
        "openrouter": "tldw_chatbook.Image_Generation.adapters.openrouter_image_adapter.OpenRouterImageAdapter",
        "novita": "tldw_chatbook.Image_Generation.adapters.novita_image_adapter.NovitaImageAdapter",
        "together": "tldw_chatbook.Image_Generation.adapters.together_image_adapter.TogetherImageAdapter",
        "modelstudio": "tldw_chatbook.Image_Generation.adapters.modelstudio_image_adapter.ModelStudioImageAdapter",
        "gemini": "tldw_chatbook.Image_Generation.adapters.gemini_image_adapter.GeminiImageAdapter",
        "fal": "tldw_chatbook.Image_Generation.adapters.fal_image_adapter.FalImageAdapter",
        "comfyui": "tldw_chatbook.Image_Generation.adapters.comfyui_image_adapter.ComfyUIImageAdapter",
    }

    def _load_config(self) -> Any:
        return get_image_generation_config()

    def _config_default_backend(self, config: Any) -> str | None:
        return config.default_backend

    def _config_enabled_backends(self, config: Any) -> list[str]:
        return config.enabled_backends

    @contextmanager
    def _config_snapshot(self) -> Iterator[None]:
        with _use_image_generation_config_snapshot(self.config):
            yield


# Export DEFAULT_ADAPTERS at module level for testing and introspection
DEFAULT_ADAPTERS = ImageAdapterRegistry.DEFAULT_ADAPTERS

_registry: ImageAdapterRegistry | None = None


def get_registry() -> ImageAdapterRegistry:
    global _registry
    with _IMAGE_GENERATION_RUNTIME_LOCK:
        if _registry is None:
            _registry = ImageAdapterRegistry()
        return _registry


def reset_registry() -> None:
    global _registry
    with _IMAGE_GENERATION_RUNTIME_LOCK:
        _registry = None
