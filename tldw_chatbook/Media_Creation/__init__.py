# Media_Creation/__init__.py
# Description: Media creation module for generating images, audio, and other media content

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .image_generation_service import ImageGenerationService
    from .swarmui_client import SwarmUIClient

from .generation_templates import (
    BUILTIN_TEMPLATES,
    GenerationTemplate,
    apply_template_to_prompt,
    get_all_categories,
    get_all_templates,
    get_template,
    get_templates_by_category,
    get_templates_by_tag,
    load_user_templates,
    reset_templates_cache,
)

__all__ = [
    "SwarmUIClient",
    "ImageGenerationService",
    "GenerationTemplate",
    "BUILTIN_TEMPLATES",
    "get_template",
    "get_templates_by_category",
    "get_all_categories",
    "get_templates_by_tag",
    "apply_template_to_prompt",
    "get_all_templates",
    "load_user_templates",
    "reset_templates_cache",
]


def __getattr__(name: str):
    """Load provider clients only when their public export is requested."""
    if name == "SwarmUIClient":
        from .swarmui_client import SwarmUIClient

        return SwarmUIClient
    if name == "ImageGenerationService":
        from .image_generation_service import ImageGenerationService

        return ImageGenerationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
