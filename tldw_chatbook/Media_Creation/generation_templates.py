# generation_templates.py
# Description: Pre-defined templates for image generation, plus user-defined
# style templates layered on top (Task-559 AC4).
#
# User templates are loaded from two sources and merged over the 13
# `BUILTIN_TEMPLATES` (a user template with the same ``id`` as a builtin
# OVERRIDES it; new ids extend the set):
#
#   1. The ``[image_generation.styles.<id>]`` TOML config section -- nested
#      sub-tables under ``[image_generation]``, following the same pattern
#      as the backend sections (e.g. ``[image_generation.swarmui]``).
#   2. One ``*.toml`` file per template under
#      ``<user_data_dir>/image_generation_styles/`` (mirrors
#      ``get_user_data_dir() / "chat_dicts"`` and ``.../"rag_profiles"`` --
#      the established convention for a user-writable per-item directory
#      under the app's data dir). The FILENAME STEM is the template's id
#      (never a same-named field inside the file, if present) -- this keeps
#      id resolution a pure function of "which file", with no risk of a
#      hand-edited internal field spoofing another template's id.
#
# Directory templates take precedence over config-section templates when
# both define the same id (a standalone file is the more deliberate,
# easier-to-share edit). Malformed entries from either source are skipped
# with a logged warning -- they must never crash the `@style` resolver or
# the Console style picker (untrusted user input).
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:  # pragma: no cover - the project requires Python >=3.11, this is belt-and-braces
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

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
    "portrait_realistic": GenerationTemplate(
        id="portrait_realistic",
        name="Realistic Portrait",
        category="Portrait",
        description="Generate a realistic portrait photo",
        base_prompt="professional portrait photo of {{subject}}, detailed face, natural lighting, high quality, 8k uhd",
        negative_prompt="cartoon, anime, drawing, painting, blurry, low quality, bad anatomy",
        default_params={
            "width": 768,
            "height": 1024,
            "steps": 30,
            "cfg_scale": 7.0,
            "sampler": "dpmpp_2m_sde",
        },
        context_mappings={"subject": "last_message", "mood": "mood"},
        tags=["portrait", "realistic", "photo"],
    ),
    "portrait_artistic": GenerationTemplate(
        id="portrait_artistic",
        name="Artistic Portrait",
        category="Portrait",
        description="Generate an artistic portrait illustration",
        base_prompt="artistic portrait of {{subject}}, digital painting, dramatic lighting, artstation quality",
        negative_prompt="photo, realistic, blurry, low quality",
        default_params={"width": 768, "height": 1024, "steps": 25, "cfg_scale": 8.0},
        context_mappings={"subject": "last_message"},
        tags=["portrait", "artistic", "illustration"],
    ),
    # Landscape templates
    "landscape_natural": GenerationTemplate(
        id="landscape_natural",
        name="Natural Landscape",
        category="Landscape",
        description="Generate a natural landscape scene",
        base_prompt="beautiful {{scene}} landscape, nature photography, golden hour, high detail, 8k",
        negative_prompt="people, buildings, text, watermark, low quality",
        default_params={"width": 1344, "height": 768, "steps": 25, "cfg_scale": 7.5},
        context_mappings={"scene": "last_message"},
        tags=["landscape", "nature", "scenic"],
    ),
    "landscape_fantasy": GenerationTemplate(
        id="landscape_fantasy",
        name="Fantasy Landscape",
        category="Landscape",
        description="Generate a fantasy landscape scene",
        base_prompt="epic fantasy landscape, {{scene}}, magical atmosphere, concept art, detailed, vibrant colors",
        negative_prompt="photo, realistic, modern, mundane, low quality",
        default_params={"width": 1344, "height": 768, "steps": 30, "cfg_scale": 8.5},
        context_mappings={"scene": "last_message"},
        tags=["landscape", "fantasy", "concept art"],
    ),
    # Concept Art templates
    "concept_character": GenerationTemplate(
        id="concept_character",
        name="Character Concept",
        category="Concept Art",
        description="Generate character concept art",
        base_prompt="character concept art of {{character}}, full body, detailed design, professional artwork",
        negative_prompt="photo, blurry, low quality, amateur",
        default_params={"width": 768, "height": 1152, "steps": 30, "cfg_scale": 8.0},
        context_mappings={"character": "last_message"},
        tags=["concept", "character", "design"],
    ),
    "concept_environment": GenerationTemplate(
        id="concept_environment",
        name="Environment Concept",
        category="Concept Art",
        description="Generate environment concept art",
        base_prompt="environment concept art, {{setting}}, atmospheric, detailed architecture, professional",
        negative_prompt="photo, people, text, low quality",
        default_params={"width": 1344, "height": 768, "steps": 30, "cfg_scale": 7.5},
        context_mappings={"setting": "last_message"},
        tags=["concept", "environment", "architecture"],
    ),
    # Style templates
    "style_anime": GenerationTemplate(
        id="style_anime",
        name="Anime Style",
        category="Style",
        description="Generate in anime/manga style",
        base_prompt="{{subject}}, anime style, detailed, vibrant colors, high quality anime art",
        negative_prompt="realistic, photo, 3d, western cartoon, low quality",
        default_params={"width": 768, "height": 1024, "steps": 25, "cfg_scale": 9.0},
        context_mappings={"subject": "last_message"},
        tags=["anime", "manga", "style"],
    ),
    "style_watercolor": GenerationTemplate(
        id="style_watercolor",
        name="Watercolor Style",
        category="Style",
        description="Generate in watercolor painting style",
        base_prompt="{{subject}}, watercolor painting, soft colors, artistic, traditional media",
        negative_prompt="photo, digital, 3d, sharp lines, low quality",
        default_params={"width": 1024, "height": 1024, "steps": 25, "cfg_scale": 7.0},
        context_mappings={"subject": "last_message"},
        tags=["watercolor", "painting", "traditional"],
    ),
    "style_cyberpunk": GenerationTemplate(
        id="style_cyberpunk",
        name="Cyberpunk Style",
        category="Style",
        description="Generate in cyberpunk aesthetic",
        base_prompt="{{subject}}, cyberpunk style, neon lights, futuristic, high tech, night scene",
        negative_prompt="medieval, rustic, natural, low tech, low quality",
        default_params={"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 8.0},
        context_mappings={"subject": "last_message"},
        tags=["cyberpunk", "futuristic", "neon"],
    ),
    # Quick generation templates
    "quick_simple": GenerationTemplate(
        id="quick_simple",
        name="Quick Simple",
        category="Quick",
        description="Fast generation with basic settings",
        base_prompt="{{prompt}}",
        negative_prompt="low quality",
        default_params={"width": 512, "height": 512, "steps": 15, "cfg_scale": 7.0},
        context_mappings={"prompt": "last_message"},
        tags=["quick", "fast", "simple"],
    ),
    "quick_quality": GenerationTemplate(
        id="quick_quality",
        name="Quick Quality",
        category="Quick",
        description="Balanced speed and quality",
        base_prompt="{{prompt}}, high quality, detailed",
        negative_prompt="blurry, low quality, amateur",
        default_params={"width": 768, "height": 768, "steps": 20, "cfg_scale": 7.5},
        context_mappings={"prompt": "last_message"},
        tags=["quick", "balanced"],
    ),
    # Chat-specific templates
    "chat_character_visual": GenerationTemplate(
        id="chat_character_visual",
        name="Character Visualization",
        category="Chat",
        description="Visualize a character from chat",
        base_prompt="character portrait of {{character_description}}, detailed, expressive",
        negative_prompt="blurry, low quality, bad anatomy",
        default_params={"width": 768, "height": 1024, "steps": 25, "cfg_scale": 7.5},
        context_mappings={"character_description": "last_message"},
        tags=["chat", "character", "visualization"],
    ),
    "chat_scene_visual": GenerationTemplate(
        id="chat_scene_visual",
        name="Scene Visualization",
        category="Chat",
        description="Visualize a scene from chat",
        base_prompt="scene depicting {{scene_description}}, atmospheric, detailed environment",
        negative_prompt="blurry, low quality, text",
        default_params={"width": 1024, "height": 768, "steps": 25, "cfg_scale": 7.5},
        context_mappings={"scene_description": "last_message"},
        tags=["chat", "scene", "visualization"],
    ),
}


# ---------------------------------------------------------------------------
# User-defined templates (Task-559 AC4): config section + templates dir,
# merged over BUILTIN_TEMPLATES. See the module docstring for the precedence
# rule and the two source shapes.
# ---------------------------------------------------------------------------

USER_TEMPLATES_DIR_NAME = "image_generation_styles"
"""Subdirectory of ``get_user_data_dir()`` holding one ``*.toml`` file per
user-defined style template, filename stem == template id."""

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_REQUIRED_STRING_FIELDS = ("name", "category", "base_prompt")


def _read_style_config_section() -> Dict[str, Any]:
    """Return the raw ``[image_generation.styles]`` table (id -> fields).

    Patch point in tests. Never raises: any failure to load settings yields
    an empty mapping (equivalent to "no user templates configured here").
    """
    try:
        from tldw_chatbook.config import load_settings

        raw = (load_settings().get("image_generation", {}) or {}).get("styles", {})
    except Exception as e:  # pragma: no cover - defensive; load_settings is well-tested elsewhere
        logger.warning(f"Failed to read [image_generation.styles] config section: {e}")
        return {}
    return raw if isinstance(raw, dict) else {}


def _user_templates_dir() -> Path:
    """``<user_data_dir>/image_generation_styles`` -- patch point in tests."""
    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / USER_TEMPLATES_DIR_NAME


def _validate_template_id(candidate: str, *, source: str) -> Optional[str]:
    """Return ``candidate`` unchanged if it's a legal template id, else ``None``.

    Legal ids are 1-64 chars of ASCII letters/digits/underscore/hyphen --
    the same shape as an existing `GenerationTemplate.id` (see
    ``BUILTIN_TEMPLATES``) and Textual DOM-id-safe (used verbatim as a
    picker row id suffix).
    """
    if isinstance(candidate, str) and _ID_PATTERN.match(candidate):
        return candidate
    logger.warning(
        f"Skipping style template with invalid id {candidate!r} from {source}"
    )
    return None


def _coerce_generation_template(
    template_id: str, data: Any, *, source: str
) -> Optional[GenerationTemplate]:
    """Validate + coerce one user-supplied style-template record.

    Mirrors `GenerationTemplate`'s field shape exactly. Required:
    ``name``/``category``/``base_prompt`` (non-empty strings). Everything
    else is optional and falls back to a sane default -- including
    ``negative_prompt``, which built-ins always set explicitly but the
    dataclass itself defaults.

    Never raises: any structural problem (not a table, missing/empty
    required field, wrong type) is logged as a warning naming ``source`` and
    ``template_id``, and ``None`` is returned so the caller skips the entry
    -- malformed user input must never crash the picker or the `@style`
    resolver.

    Args:
        template_id: Already-validated id (see `_validate_template_id`).
        data: The raw parsed record (expected to be a TOML table/dict).
        source: Human-readable origin for the warning message (e.g. a
            config path or a templates-dir filename).

    Returns:
        A `GenerationTemplate`, or ``None`` when `data` fails validation.
    """
    if not isinstance(data, dict):
        logger.warning(
            f"Skipping malformed style template '{template_id}' from {source}: "
            f"expected a table, got {type(data).__name__}"
        )
        return None

    for field_name in _REQUIRED_STRING_FIELDS:
        value = data.get(field_name)
        if not isinstance(value, str) or not value.strip():
            logger.warning(
                f"Skipping malformed style template '{template_id}' from {source}: "
                f"missing or empty required field '{field_name}'"
            )
            return None

    negative_prompt = data.get("negative_prompt")
    if not isinstance(negative_prompt, str) or not negative_prompt.strip():
        negative_prompt = GenerationTemplate.__dataclass_fields__[
            "negative_prompt"
        ].default

    description = data.get("description")
    if not isinstance(description, str):
        description = ""

    default_params_raw = data.get("default_params")
    default_params = dict(default_params_raw) if isinstance(default_params_raw, dict) else {}

    context_mappings_raw = data.get("context_mappings")
    context_mappings: Dict[str, str] = {}
    if isinstance(context_mappings_raw, dict):
        context_mappings = {
            str(k): str(v)
            for k, v in context_mappings_raw.items()
            if isinstance(k, str) and isinstance(v, str)
        }

    tags_raw = data.get("tags")
    tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []

    return GenerationTemplate(
        id=template_id,
        name=data["name"].strip(),
        category=data["category"].strip(),
        description=description,
        base_prompt=data["base_prompt"],
        negative_prompt=negative_prompt,
        default_params=default_params,
        context_mappings=context_mappings,
        tags=tags,
    )


def _load_config_section_templates() -> Dict[str, GenerationTemplate]:
    """Load user templates from the ``[image_generation.styles]`` config section."""
    raw = _read_style_config_section()
    templates: Dict[str, GenerationTemplate] = {}
    for raw_id, data in raw.items():
        template_id = _validate_template_id(
            str(raw_id), source="[image_generation.styles] config"
        )
        if template_id is None:
            continue
        template = _coerce_generation_template(
            template_id, data, source=f"[image_generation.styles.{raw_id}] config"
        )
        if template is not None:
            templates[template_id] = template
    return templates


def _load_directory_templates() -> Dict[str, GenerationTemplate]:
    """Load user templates from ``<user_data_dir>/image_generation_styles/*.toml``."""
    directory = _user_templates_dir()
    templates: Dict[str, GenerationTemplate] = {}
    try:
        if not directory.is_dir():
            return templates
        paths = sorted(directory.glob("*.toml"))
    except OSError as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to list style templates dir {directory}: {e}")
        return templates

    for path in paths:
        template_id = _validate_template_id(
            path.stem, source=f"templates dir file {path.name!r}"
        )
        if template_id is None:
            continue
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            logger.warning(f"Skipping unparsable style template file {path.name!r}: {e}")
            continue
        template = _coerce_generation_template(
            template_id, data, source=f"templates dir file {path.name!r}"
        )
        if template is not None:
            templates[template_id] = template
    return templates


def load_user_templates() -> Dict[str, GenerationTemplate]:
    """User-defined style templates merged from config + templates dir.

    Directory-loaded templates take precedence over config-section templates
    when both define the same id -- see module docstring.

    Returns:
        A fresh dict (never mutates module state); keyed by template id.
    """
    merged = _load_config_section_templates()
    merged.update(_load_directory_templates())
    return merged


_all_templates_cache: Optional[Dict[str, GenerationTemplate]] = None


def get_all_templates(*, reload: bool = False) -> Dict[str, GenerationTemplate]:
    """`BUILTIN_TEMPLATES` merged with user-defined templates (config + dir).

    A user template with the same id as a builtin OVERRIDES it; new ids
    extend the set. This is the set every `/generate-image` `@style`
    resolution, refusal listing, and the Console/Personas style picker draws
    from -- user templates work everywhere a builtin does.

    Cached like `Image_Generation.config.get_image_generation_config` (same
    process-lifetime cache); pass ``reload=True`` or call
    `reset_templates_cache` after templates change on disk/in config.

    Args:
        reload: Force a fresh reload, bypassing the cache.

    Returns:
        A dict of every available template, keyed by id. The caller must
        treat it as read-only -- it may be the live cached instance.
    """
    global _all_templates_cache
    if _all_templates_cache is not None and not reload:
        return _all_templates_cache
    merged: Dict[str, GenerationTemplate] = dict(BUILTIN_TEMPLATES)
    merged.update(load_user_templates())
    _all_templates_cache = merged
    return merged


def reset_templates_cache() -> None:
    """Clear `get_all_templates`'s cache (test/reload seam)."""
    global _all_templates_cache
    _all_templates_cache = None


def get_template(template_id: str) -> Optional[GenerationTemplate]:
    """Get a template by ID (builtin or user-defined; see `get_all_templates`).

    Args:
        template_id: Template identifier

    Returns:
        Template if found, None otherwise
    """
    template = get_all_templates().get(template_id)
    if template:
        logger.debug(f"Retrieved template: {template_id}")
    else:
        logger.warning(f"Template not found: {template_id}")
    return template


def get_templates_by_category(category: str) -> List[GenerationTemplate]:
    """Get all templates in a category (builtin or user-defined).

    Args:
        category: Category name

    Returns:
        List of templates in the category
    """
    templates = [t for t in get_all_templates().values() if t.category == category]
    logger.debug(f"Found {len(templates)} templates in category: {category}")
    return templates


def get_all_categories() -> List[str]:
    """Get list of all template categories (builtin or user-defined).

    Returns:
        List of unique category names
    """
    categories = list(set(t.category for t in get_all_templates().values()))
    categories.sort()
    return categories


def get_templates_by_tag(tag: str) -> List[GenerationTemplate]:
    """Get all templates with a specific tag (builtin or user-defined).

    Args:
        tag: Tag to search for

    Returns:
        List of templates with the tag
    """
    templates = [t for t in get_all_templates().values() if tag in t.tags]
    logger.debug(f"Found {len(templates)} templates with tag: {tag}")
    return templates


def apply_template_to_prompt(
    template_id: str, context: Dict[str, Any]
) -> Tuple[str, str, Dict[str, Any]]:
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
    prompt = re.sub(r"\{\{[^}]+\}\}", "", prompt).strip()

    return prompt, template.negative_prompt, template.default_params.copy()
