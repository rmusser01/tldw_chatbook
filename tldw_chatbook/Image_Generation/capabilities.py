"""Shared capability resolution for image generation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Image_Generation.config import ImageGenerationConfig

_DEFAULT_REFERENCE_IMAGE_SUPPORT: dict[str, set[str]] = {
    "modelstudio": {"qwen-image-2.0", "qwen-image-edit"},
}


@dataclass(frozen=True)
class ResolvedReferenceImage:
    """Normalized reference image payload used by image generation requests."""

    file_id: int | str
    filename: str | None
    mime_type: str
    width: int | None
    height: int | None
    bytes_len: int
    content: bytes | None
    temp_path: str | None

    def __post_init__(self) -> None:
        has_content = self.content is not None
        has_temp_path = self.temp_path is not None
        if has_content == has_temp_path:
            raise ValueError("ResolvedReferenceImage requires exactly one of content or temp_path")


@dataclass(frozen=True)
class ReferenceImageCapability:
    """Resolved image reference support for a backend/model pair."""

    supported: bool
    reason: str | None = None


def _normalize_key(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_model_family(value: str | None) -> str:
    return _normalize_key(value).rstrip("*")


def _matches_model_family(family_root: str, model: str) -> bool:
    if not family_root or not model:
        return False
    if model == family_root:
        return True
    for separator in ("-", ".", "_", "/"):
        if model.startswith(f"{family_root}{separator}"):
            return True
    return False


def _normalize_model_map(value: dict[str, list[str]] | None) -> dict[str, set[str]]:
    if not value:
        return {}
    normalized: dict[str, set[str]] = {}
    for backend, models in value.items():
        backend_key = _normalize_key(backend)
        if not backend_key:
            continue
        normalized[backend_key] = {_normalize_model_family(model) for model in models if _normalize_model_family(model)}
    return normalized


def _resolve_supported_models(
    backend_key: str,
    *,
    config: ImageGenerationConfig | None = None,
) -> set[str]:
    builtin_support = _DEFAULT_REFERENCE_IMAGE_SUPPORT.get(backend_key, set())
    if not builtin_support:
        return set()

    configured_support = _normalize_model_map(
        getattr(config, "reference_image_supported_models", None) if config is not None else None
    )
    configured_models = configured_support.get(backend_key)
    if configured_models is None:
        return set(builtin_support)
    return {
        model
        for model in configured_models
        if any(_matches_model_family(builtin_model, model) for builtin_model in builtin_support)
    }


def resolve_backend_reference_image_capability(
    backend: str | None,
    *,
    config: ImageGenerationConfig | None = None,
) -> ReferenceImageCapability:
    """Resolve whether a backend exposes any reference-image support."""

    backend_key = _normalize_key(backend)
    if not backend_key:
        return ReferenceImageCapability(supported=False, reason="unsupported_model")
    supported_models = _resolve_supported_models(backend_key, config=config)
    return ReferenceImageCapability(
        supported=bool(supported_models),
        reason=None if supported_models else "unsupported_model",
    )


def resolve_reference_image_capability(
    backend: str | None,
    model: str | None,
    *,
    config: ImageGenerationConfig | None = None,
) -> ReferenceImageCapability:
    """Resolve whether a backend/model pair supports reference-image input."""

    backend_key = _normalize_key(backend)
    model_key = _normalize_key(model)
    if not backend_key or not model_key:
        return ReferenceImageCapability(supported=False, reason="unsupported_model")

    supported_models = _resolve_supported_models(backend_key, config=config)
    if not supported_models:
        return ReferenceImageCapability(supported=False, reason="unsupported_model")

    if any(_matches_model_family(supported_model, model_key) for supported_model in supported_models):
        return ReferenceImageCapability(supported=True)
    return ReferenceImageCapability(supported=False, reason="unsupported_model")
