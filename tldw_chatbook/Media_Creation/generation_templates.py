# generation_templates.py
# Description: Pre-defined templates for image generation

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger


@dataclass
class GenerationTemplate:
    """Template for image generation with pre-configured settings."""
    id: str
    name: str
    category: str
    description: str
    base_prompt: str
    negative_prompt: str = "blurry, low quality, bad anatomy, ugly, deformed"
    default_params: Dict[str, Any] = field(default_factory=dict)
    context_mappings: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


# Built-in templates
BUILTIN_TEMPLATES = {
    # Portrait templates
    'portrait_realistic': GenerationTemplate(
        id='portrait_realistic',
        name='Realistic Portrait',
        category='Portrait',
        description='Generate a realistic portrait photo',
        base_prompt='professional portrait photo of {{subject}}, detailed face, natural lighting, high quality, 8k uhd',
        negative_prompt='cartoon, anime, drawing, painting, blurry, low quality, bad anatomy',
        default_params={
            'width': 768,
            'height': 1024,
            'steps': 30,
            'cfg_scale': 7.0,
            'sampler': 'dpmpp_2m_sde'
        },
        context_mappings={
            'subject': 'last_message',
            'mood': 'mood'
        },
        tags=['portrait', 'realistic', 'photo']
    ),
    
    'portrait_artistic': GenerationTemplate(
        id='portrait_artistic',
        name='Artistic Portrait',
        category='Portrait',
        description='Generate an artistic portrait illustration',
        base_prompt='artistic portrait of {{subject}}, digital painting, dramatic lighting, artstation quality',
        negative_prompt='photo, realistic, blurry, low quality',
        default_params={
            'width': 768,
            'height': 1024,
            'steps': 25,
            'cfg_scale': 8.0
        },
        context_mappings={
            'subject': 'last_message'
        },
        tags=['portrait', 'artistic', 'illustration']
    ),
    
    # Landscape templates
    'landscape_natural': GenerationTemplate(
        id='landscape_natural',
        name='Natural Landscape',
        category='Landscape',
        description='Generate a natural landscape scene',
        base_prompt='beautiful {{scene}} landscape, nature photography, golden hour, high detail, 8k',
        negative_prompt='people, buildings, text, watermark, low quality',
        default_params={
            'width': 1344,
            'height': 768,
            'steps': 25,
            'cfg_scale': 7.5
        },
        context_mappings={
            'scene': 'last_message'
        },
        tags=['landscape', 'nature', 'scenic']
    ),
    
    'landscape_fantasy': GenerationTemplate(
        id='landscape_fantasy',
        name='Fantasy Landscape',
        category='Landscape',
        description='Generate a fantasy landscape scene',
        base_prompt='epic fantasy landscape, {{scene}}, magical atmosphere, concept art, detailed, vibrant colors',
        negative_prompt='photo, realistic, modern, mundane, low quality',
        default_params={
            'width': 1344,
            'height': 768,
            'steps': 30,
            'cfg_scale': 8.5
        },
        context_mappings={
            'scene': 'last_message'
        },
        tags=['landscape', 'fantasy', 'concept art']
    ),
    
    # Concept Art templates
    'concept_character': GenerationTemplate(
        id='concept_character',
        name='Character Concept',
        category='Concept Art',
        description='Generate character concept art',
        base_prompt='character concept art of {{character}}, full body, detailed design, professional artwork',
        negative_prompt='photo, blurry, low quality, amateur',
        default_params={
            'width': 768,
            'height': 1152,
            'steps': 30,
            'cfg_scale': 8.0
        },
        context_mappings={
            'character': 'last_message'
        },
        tags=['concept', 'character', 'design']
    ),
    
    'concept_environment': GenerationTemplate(
        id='concept_environment',
        name='Environment Concept',
        category='Concept Art',
        description='Generate environment concept art',
        base_prompt='environment concept art, {{setting}}, atmospheric, detailed architecture, professional',
        negative_prompt='photo, people, text, low quality',
        default_params={
            'width': 1344,
            'height': 768,
            'steps': 30,
            'cfg_scale': 7.5
        },
        context_mappings={
            'setting': 'last_message'
        },
        tags=['concept', 'environment', 'architecture']
    ),
    
    # Style templates
    'style_anime': GenerationTemplate(
        id='style_anime',
        name='Anime Style',
        category='Style',
        description='Generate in anime/manga style',
        base_prompt='{{subject}}, anime style, detailed, vibrant colors, high quality anime art',
        negative_prompt='realistic, photo, 3d, western cartoon, low quality',
        default_params={
            'width': 768,
            'height': 1024,
            'steps': 25,
            'cfg_scale': 9.0
        },
        context_mappings={
            'subject': 'last_message'
        },
        tags=['anime', 'manga', 'style']
    ),
    
    'style_watercolor': GenerationTemplate(
        id='style_watercolor',
        name='Watercolor Style',
        category='Style',
        description='Generate in watercolor painting style',
        base_prompt='{{subject}}, watercolor painting, soft colors, artistic, traditional media',
        negative_prompt='photo, digital, 3d, sharp lines, low quality',
        default_params={
            'width': 1024,
            'height': 1024,
            'steps': 25,
            'cfg_scale': 7.0
        },
        context_mappings={
            'subject': 'last_message'
        },
        tags=['watercolor', 'painting', 'traditional']
    ),
    
    'style_cyberpunk': GenerationTemplate(
        id='style_cyberpunk',
        name='Cyberpunk Style',
        category='Style',
        description='Generate in cyberpunk aesthetic',
        base_prompt='{{subject}}, cyberpunk style, neon lights, futuristic, high tech, night scene',
        negative_prompt='medieval, rustic, natural, low tech, low quality',
        default_params={
            'width': 1024,
            'height': 1024,
            'steps': 30,
            'cfg_scale': 8.0
        },
        context_mappings={
            'subject': 'last_message'
        },
        tags=['cyberpunk', 'futuristic', 'neon']
    ),
    
    # Quick generation templates
    'quick_simple': GenerationTemplate(
        id='quick_simple',
        name='Quick Simple',
        category='Quick',
        description='Fast generation with basic settings',
        base_prompt='{{prompt}}',
        negative_prompt='low quality',
        default_params={
            'width': 512,
            'height': 512,
            'steps': 15,
            'cfg_scale': 7.0
        },
        context_mappings={
            'prompt': 'last_message'
        },
        tags=['quick', 'fast', 'simple']
    ),
    
    'quick_quality': GenerationTemplate(
        id='quick_quality',
        name='Quick Quality',
        category='Quick',
        description='Balanced speed and quality',
        base_prompt='{{prompt}}, high quality, detailed',
        negative_prompt='blurry, low quality, amateur',
        default_params={
            'width': 768,
            'height': 768,
            'steps': 20,
            'cfg_scale': 7.5
        },
        context_mappings={
            'prompt': 'last_message'
        },
        tags=['quick', 'balanced']
    ),
    
    # Chat-specific templates
    'chat_character_visual': GenerationTemplate(
        id='chat_character_visual',
        name='Character Visualization',
        category='Chat',
        description='Visualize a character from chat',
        base_prompt='character portrait of {{character_description}}, detailed, expressive',
        negative_prompt='blurry, low quality, bad anatomy',
        default_params={
            'width': 768,
            'height': 1024,
            'steps': 25,
            'cfg_scale': 7.5
        },
        context_mappings={
            'character_description': 'last_message'
        },
        tags=['chat', 'character', 'visualization']
    ),
    
    'chat_scene_visual': GenerationTemplate(
        id='chat_scene_visual',
        name='Scene Visualization',
        category='Chat',
        description='Visualize a scene from chat',
        base_prompt='scene depicting {{scene_description}}, atmospheric, detailed environment',
        negative_prompt='blurry, low quality, text',
        default_params={
            'width': 1024,
            'height': 768,
            'steps': 25,
            'cfg_scale': 7.5
        },
        context_mappings={
            'scene_description': 'last_message'
        },
        tags=['chat', 'scene', 'visualization']
    )
}


def get_template(template_id: str) -> Optional[GenerationTemplate]:
    """Get a template by ID.
    
    Args:
        template_id: Template identifier
        
    Returns:
        Template if found, None otherwise
    """
    template = BUILTIN_TEMPLATES.get(template_id)
    if template:
        logger.debug(f"Retrieved template: {template_id}")
    else:
        logger.warning(f"Template not found: {template_id}")
    return template


def get_templates_by_category(category: str) -> List[GenerationTemplate]:
    """Get all templates in a category.
    
    Args:
        category: Category name
        
    Returns:
        List of templates in the category
    """
    templates = [t for t in BUILTIN_TEMPLATES.values() if t.category == category]
    logger.debug(f"Found {len(templates)} templates in category: {category}")
    return templates


def get_all_categories() -> List[str]:
    """Get list of all template categories.
    
    Returns:
        List of unique category names
    """
    categories = list(set(t.category for t in BUILTIN_TEMPLATES.values()))
    categories.sort()
    return categories


def get_templates_by_tag(tag: str) -> List[GenerationTemplate]:
    """Get all templates with a specific tag.
    
    Args:
        tag: Tag to search for
        
    Returns:
        List of templates with the tag
    """
    templates = [t for t in BUILTIN_TEMPLATES.values() if tag in t.tags]
    logger.debug(f"Found {len(templates)} templates with tag: {tag}")
    return templates


def apply_template_to_prompt(template_id: str, context: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Apply a template with context to generate final prompt and parameters.
    
    Args:
        template_id: Template to use
        context: Context dictionary with values for template variables
        
    Returns:
        Tuple of (prompt, negative_prompt, parameters)
    """
    template = get_template(template_id)
    if not template:
        return "", "", {}
    
    prompt = template.base_prompt
    
    # Apply context mappings
    for key, mapping in template.context_mappings.items():
        if mapping in context and context[mapping]:
            placeholder = f"{{{{{key}}}}}"
            value = str(context[mapping])
            prompt = prompt.replace(placeholder, value)
    
    # Remove any remaining placeholders
    import re
    prompt = re.sub(r'\{\{[^}]+\}\}', '', prompt).strip()
    
    return prompt, template.negative_prompt, template.default_params.copy()