"""Registry for video generation backends (skeleton in Media_Generation)."""

from __future__ import annotations

import threading
from typing import Any

from tldw_chatbook.Media_Generation.adapter_registry import MediaAdapterRegistry
from tldw_chatbook.Video_Generation.adapters.base import VideoGenerationAdapter
from tldw_chatbook.Video_Generation.config import get_video_generation_config


class VideoAdapterRegistry(MediaAdapterRegistry):
    """Registry for video generation adapters.

    The three DEFAULT_ADAPTERS specs point at the classes their backend
    tasks (3401.3/.6/.7) will provide. Resolution is lazy: ``resolve_backend``
    never imports, and ``get_adapter`` logs + returns ``None`` when the class
    is not importable yet -- so enabling a not-yet-shipped backend fails
    cleanly at generation time rather than at import time.

    ADR-176: the shared skeleton lives in ``Media_Generation.adapter_registry``.
    Registry get/reset now hold a runtime lock (image's hardening applied to
    both modalities); adapter-failure logs keep video's privacy-stricter
    error-type-only detail. Video has no config-snapshot machinery yet, so
    adapters construct without a snapshot (adding one is a recorded
    follow-up requiring video config-snapshot support).
    """

    modality = "video"
    log_error_detail = False

    DEFAULT_ADAPTERS: dict[str, str] = {
        "minimax": "tldw_chatbook.Video_Generation.adapters.minimax_video_adapter.MiniMaxVideoAdapter",
        "comfyui": "tldw_chatbook.Video_Generation.adapters.comfyui_video_adapter.ComfyUIVideoAdapter",
        "stable_diffusion_cpp": "tldw_chatbook.Video_Generation.adapters.stable_diffusion_cpp_video_adapter.StableDiffusionCppVideoAdapter",
    }

    def _load_config(self) -> Any:
        return get_video_generation_config()

    def _config_default_backend(self, config: Any) -> str | None:
        return config.default_backend

    def _config_enabled_backends(self, config: Any) -> list[str]:
        return config.enabled_backends


# Export DEFAULT_ADAPTERS at module level for testing and introspection
DEFAULT_ADAPTERS = VideoAdapterRegistry.DEFAULT_ADAPTERS

_registry: VideoAdapterRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> VideoAdapterRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = VideoAdapterRegistry()
        return _registry


def reset_registry() -> None:
    global _registry
    with _registry_lock:
        _registry = None
