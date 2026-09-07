"""Registry for video generation backends."""

from __future__ import annotations

import importlib
import json
from typing import Any

from loguru import logger

from tldw_chatbook.Video_Generation.adapters.base import VideoGenerationAdapter
from tldw_chatbook.Video_Generation.config import get_video_generation_config


class VideoAdapterRegistry:
    """Registry for video generation adapters.

    The three DEFAULT_ADAPTERS specs point at the classes their backend
    tasks (3401.3/.6/.7) will provide. Resolution is lazy: ``resolve_backend``
    never imports, and ``get_adapter`` logs + returns ``None`` when the class
    is not importable yet -- so enabling a not-yet-shipped backend fails
    cleanly at generation time rather than at import time.
    """

    DEFAULT_ADAPTERS: dict[str, str] = {
        "minimax": "tldw_chatbook.Video_Generation.adapters.minimax_video_adapter.MiniMaxVideoAdapter",
        "comfyui": "tldw_chatbook.Video_Generation.adapters.comfyui_video_adapter.ComfyUIVideoAdapter",
        "stable_diffusion_cpp": "tldw_chatbook.Video_Generation.adapters.stable_diffusion_cpp_video_adapter.StableDiffusionCppVideoAdapter",
    }

    def __init__(self, config_override: dict[str, Any] | None = None) -> None:
        config = get_video_generation_config()
        default_backend = config.default_backend
        enabled_backends = list(config.enabled_backends)
        if config_override:
            if "default_backend" in config_override:
                default_backend = str(config_override.get("default_backend") or "").strip() or None
            if "enabled_backends" in config_override:
                enabled_backends = self._parse_list(config_override.get("enabled_backends"))
        self._default_backend = default_backend
        self._enabled_backends = enabled_backends
        self._adapters: dict[str, VideoGenerationAdapter] = {}
        self._adapter_specs: dict[str, Any] = self.DEFAULT_ADAPTERS.copy()

    def register_adapter(self, name: str, adapter: Any) -> None:
        self._adapter_specs[name] = adapter
        try:
            adapter_name = adapter.__name__  # type: ignore[attr-defined]
        except Exception:
            adapter_name = str(adapter)
        logger.info("Registered video adapter {} for backend '{}'", adapter_name, name)

    def list_backend_names(self, *, include_disabled: bool = False) -> list[str]:
        names = list(self._adapter_specs.keys())
        if include_disabled:
            return names
        if not self._enabled_backends:
            return []
        return [name for name in names if name in self._enabled_backends]

    def _resolve_adapter_class(self, spec: Any) -> type[VideoGenerationAdapter]:
        if isinstance(spec, str):
            module_path, _, class_name = spec.rpartition(".")
            if not module_path:
                raise ImportError(f"Invalid adapter spec '{spec}'")
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        return spec

    def _is_enabled(self, name: str) -> bool:
        if not self._enabled_backends:
            return False
        return name in self._enabled_backends

    def resolve_backend(self, requested: str | None) -> str | None:
        name = (requested or self._default_backend or "").strip()
        if not name:
            return None
        if not self._is_enabled(name):
            return None
        if name not in self._adapter_specs:
            return None
        return name

    def get_adapter(self, name: str) -> VideoGenerationAdapter | None:
        if name in self._adapters:
            return self._adapters[name]

        spec = self._adapter_specs.get(name)
        if not spec:
            logger.debug("No video adapter spec registered for backend '{}'", name)
            return None

        try:
            adapter_cls = self._resolve_adapter_class(spec)
            adapter = adapter_cls()  # type: ignore[call-arg]
            self._adapters[name] = adapter
            return adapter
        except Exception as exc:
            logger.error(
                "Failed to initialize video adapter for '{}' (error_type={})",
                name,
                type(exc).__name__,
            )
            return None

    def get_adapter_class(self, name: str) -> type[VideoGenerationAdapter] | None:
        spec = self._adapter_specs.get(name)
        if not spec:
            logger.debug("No video adapter spec registered for backend '{}'", name)
            return None
        try:
            return self._resolve_adapter_class(spec)
        except Exception as exc:
            logger.error(
                "Failed to resolve video adapter class for '{}' (error_type={})",
                name,
                type(exc).__name__,
            )
            return None

    @staticmethod
    def _parse_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raw = str(value).strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in raw.split(",") if item.strip()]


# Export DEFAULT_ADAPTERS at module level for testing and introspection
DEFAULT_ADAPTERS = VideoAdapterRegistry.DEFAULT_ADAPTERS

_registry: VideoAdapterRegistry | None = None


def get_registry() -> VideoAdapterRegistry:
    global _registry
    if _registry is None:
        _registry = VideoAdapterRegistry()
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = None
