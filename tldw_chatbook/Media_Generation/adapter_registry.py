"""The modality-shared adapter-registry skeleton (ADR-176).

Per-modality packages subclass with their backend spec tables and config
wiring; this module owns lazy resolution, enablement, per-backend caching,
and the register/reset lifecycle. Delta resolution per ADR-176: adapter
construction runs inside the modality's config snapshot where the modality
provides one, registry get/reset hold a runtime lock for both modalities,
and adapter-failure log detail stays per-modality (image logs the
exception text; video logs only the error type, the privacy-stricter
form -- unifying it is a recorded follow-up, not a silent average).
"""

from __future__ import annotations

import importlib
import json
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

from loguru import logger


class MediaAdapterRegistry:
    """Registry skeleton shared by the image and video packages."""

    #: Singular modality name used in log wording.
    modality: str = "media"
    #: Include exception text in adapter-failure logs; when False only the
    #: error type is logged (video's privacy-stricter form).
    log_error_detail: bool = True

    DEFAULT_ADAPTERS: dict[str, str] = {}

    def __init__(self, config_override: dict[str, Any] | None = None) -> None:
        config = self._load_config()
        self.config = config
        default_backend = self._config_default_backend(config)
        enabled_backends = list(self._config_enabled_backends(config))
        if config_override:
            if "default_backend" in config_override:
                default_backend = str(config_override.get("default_backend") or "").strip() or None
            if "enabled_backends" in config_override:
                enabled_backends = self._parse_list(config_override.get("enabled_backends"))
        self._default_backend = default_backend
        self._enabled_backends = enabled_backends
        self._adapters: dict[str, Any] = {}
        self._adapter_specs: dict[str, Any] = self.DEFAULT_ADAPTERS.copy()

    # --- per-modality wiring (subclasses implement) ---

    def _load_config(self) -> Any:
        """Return the modality's generation config object."""
        raise NotImplementedError

    def _config_default_backend(self, config: Any) -> str | None:
        """Read the configured default backend from the config object."""
        raise NotImplementedError

    def _config_enabled_backends(self, config: Any) -> list[str]:
        """Read the configured enabled backends from the config object."""
        raise NotImplementedError

    @contextmanager
    def _config_snapshot(self) -> Iterator[None]:
        """Pin the config an adapter is constructed under, when supported."""
        yield

    # --- shared skeleton ---

    def register_adapter(self, name: str, adapter: Any) -> None:
        """Register an adapter spec under a backend name.

        Args:
            name: Backend name.
            adapter: Adapter class or dotted spec string.
        """
        self._adapter_specs[name] = adapter
        try:
            adapter_name = adapter.__name__  # type: ignore[attr-defined]
        except Exception:
            adapter_name = str(adapter)
        logger.info(
            "Registered {} adapter {} for backend '{}'", self.modality, adapter_name, name
        )

    def list_backend_names(self, *, include_disabled: bool = False) -> list[str]:
        names = list(self._adapter_specs.keys())
        if include_disabled:
            return names
        if not self._enabled_backends:
            return []
        return [name for name in names if name in self._enabled_backends]

    def _resolve_adapter_class(self, spec: Any) -> type:
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

    def get_adapter(self, name: str) -> Any | None:
        if name in self._adapters:
            return self._adapters[name]

        spec = self._adapter_specs.get(name)
        if not spec:
            logger.debug(
                "No {} adapter spec registered for backend '{}'", self.modality, name
            )
            return None

        try:
            adapter_cls = self._resolve_adapter_class(spec)
            with self._config_snapshot():
                adapter = adapter_cls()  # type: ignore[call-arg]
            self._adapters[name] = adapter
            return adapter
        except Exception as exc:
            if self.log_error_detail:
                logger.error(
                    "Failed to initialize {} adapter for '{}': {}",
                    self.modality,
                    name,
                    exc,
                )
            else:
                logger.error(
                    "Failed to initialize {} adapter for '{}' (error_type={})",
                    self.modality,
                    name,
                    type(exc).__name__,
                )
            return None

    def get_adapter_class(self, name: str) -> type | None:
        spec = self._adapter_specs.get(name)
        if not spec:
            logger.debug(
                "No {} adapter spec registered for backend '{}'", self.modality, name
            )
            return None
        try:
            return self._resolve_adapter_class(spec)
        except Exception as exc:
            if self.log_error_detail:
                logger.error(
                    "Failed to resolve {} adapter class for '{}': {}",
                    self.modality,
                    name,
                    exc,
                )
            else:
                logger.error(
                    "Failed to resolve {} adapter class for '{}' (error_type={})",
                    self.modality,
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
