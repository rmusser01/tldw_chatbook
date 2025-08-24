# Media_Creation/__init__.py
# Description: Media creation module for generating images, audio, and other media content

from .swarmui_client import SwarmUIClient
from .image_generation_service import ImageGenerationService
from .generation_templates import (
    GenerationTemplate, 
    BUILTIN_TEMPLATES,
    get_template,
    get_templates_by_category,
    get_all_categories,
    get_templates_by_tag,
    apply_template_to_prompt
)

__all__ = [
    'SwarmUIClient',
    'ImageGenerationService', 
    'GenerationTemplate',
    'BUILTIN_TEMPLATES',
    'get_template',
    'get_templates_by_category',
    'get_all_categories',
    'get_templates_by_tag',
    'apply_template_to_prompt'
]