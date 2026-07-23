"""Multi-provider image generation (ported from tldw_server). Import-light."""
from tldw_chatbook.Image_Generation.exceptions import (
    ImageGenerationError,
    ImageBackendUnavailableError,
)

__all__ = [
    "ImageGenerationError",
    "ImageBackendUnavailableError",
    "get_image_generation_config",
    "get_registry",
]


def __getattr__(name):  # PEP 562 lazy re-export; keeps adapters/Pillow out of import time
    if name == "get_image_generation_config":
        from tldw_chatbook.Image_Generation.config import (
            get_image_generation_config as f,
        )

        return f
    if name == "get_registry":
        from tldw_chatbook.Image_Generation.adapter_registry import get_registry as f

        return f
    raise AttributeError(name)
