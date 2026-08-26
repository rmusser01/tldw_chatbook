"""Style templates for video generation (task-3401.12, AC2).

Mirrors ``Media_Creation/generation_templates.py``'s shape but tuned for
video: a style contributes a PROMPT SUFFIX (style/camera language the video
models understand) plus default generation params (duration/fps/ratio),
rather than an image template's subject-substituted base prompt.

User templates layer over the builtins from ONE source in v1: the
``[video_generation.styles.<id>]`` TOML section (same nested-sub-table
pattern as the backend sections). A user template with a builtin's ``id``
overrides it. Malformed entries are skipped with a logged warning -- they
must never crash the ``@style`` resolver (untrusted user input). The
per-file directory convention from the image side is a deliberate v2 item
(one source of truth first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class VideoStyleTemplate:
    """One ``@style`` for video generation.

    Attributes:
        id: Token the user types after ``@`` (e.g. ``cinematic``).
        name: Display name.
        description: One-line what-it-does.
        prompt_suffix: Style/camera language appended to the user's prompt.
        negative_prompt_suffix: Appended to the negative prompt when the
            backend uses one (local backends only; MiniMax takes none).
        default_params: Generation defaults the style contributes when the
            invocation doesn't override them: ``duration_seconds``, ``fps``,
            ``ratio``.
    """

    id: str
    name: str
    description: str
    prompt_suffix: str
    negative_prompt_suffix: str = ""
    default_params: dict[str, Any] = field(default_factory=dict)


BUILTIN_VIDEO_TEMPLATES: dict[str, VideoStyleTemplate] = {
    "cinematic": VideoStyleTemplate(
        id="cinematic",
        name="Cinematic",
        description="Film-still look, dramatic light, smooth camera motion",
        prompt_suffix=(
            "cinematic film still, shallow depth of field, dramatic lighting, "
            "smooth deliberate camera motion, high detail"
        ),
        negative_prompt_suffix="flicker, jitter, low quality",
        default_params={"duration_seconds": 6, "fps": 24, "ratio": "16:9"},
    ),
    "drone": VideoStyleTemplate(
        id="drone",
        name="Drone / FPV",
        description="Sweeping aerial forward motion",
        prompt_suffix=(
            "aerial FPV drone shot, sweeping continuous forward motion, "
            "dynamic perspective, crisp horizon"
        ),
        negative_prompt_suffix="flicker, warping, low quality",
        default_params={"duration_seconds": 6, "fps": 24, "ratio": "16:9"},
    ),
    "timelapse": VideoStyleTemplate(
        id="timelapse",
        name="Timelapse",
        description="Fixed camera, accelerated motion",
        prompt_suffix=(
            "timelapse, fixed locked-off camera, accelerated motion, "
            "smooth light transitions"
        ),
        negative_prompt_suffix="camera shake, flicker, low quality",
        default_params={"duration_seconds": 8, "fps": 24, "ratio": "16:9"},
    ),
    "anime": VideoStyleTemplate(
        id="anime",
        name="Anime",
        description="Cel-shaded animation look",
        prompt_suffix=(
            "anime style, cel shading, clean line work, smooth animation, "
            "vivid color"
        ),
        negative_prompt_suffix="photorealistic, flicker, low quality",
        default_params={"duration_seconds": 5, "fps": 24, "ratio": "16:9"},
    ),
}

_STYLE_TOML_KEYS = frozenset(
    {"name", "description", "prompt_suffix", "negative_prompt_suffix", "default_params"}
)


def _user_style_tables() -> dict[str, Any]:
    """Return the raw ``[video_generation.styles]`` table (patch point in tests)."""
    from tldw_chatbook.config import load_settings

    styles = load_settings().get("video_generation", {}).get("styles", {})
    return styles if isinstance(styles, dict) else {}


def get_all_video_templates() -> dict[str, VideoStyleTemplate]:
    """Builtins overlaid with ``[video_generation.styles.<id>]`` entries."""
    templates: dict[str, VideoStyleTemplate] = dict(BUILTIN_VIDEO_TEMPLATES)
    for style_id, raw in _user_style_tables().items():
        if not isinstance(raw, dict):
            logger.warning("video style is not a table; skipped")
            continue
        unknown = set(raw) - _STYLE_TOML_KEYS
        if unknown:
            logger.warning(
                "video style has unknown keys (count={}); skipped", len(unknown)
            )
            continue
        prompt_suffix = str(raw.get("prompt_suffix") or "").strip()
        if not prompt_suffix:
            logger.warning("video style has no prompt_suffix; skipped")
            continue
        params = raw.get("default_params") or {}
        templates[str(style_id)] = VideoStyleTemplate(
            id=str(style_id),
            name=str(raw.get("name") or style_id),
            description=str(raw.get("description") or ""),
            prompt_suffix=prompt_suffix,
            negative_prompt_suffix=str(raw.get("negative_prompt_suffix") or ""),
            default_params=params if isinstance(params, dict) else {},
        )
    return templates


def get_video_template(style_id: str) -> VideoStyleTemplate | None:
    """Resolve one ``@style`` token (case-insensitive); ``None`` when unknown."""
    return get_all_video_templates().get(style_id.strip().lower())


def apply_video_template(
    template: VideoStyleTemplate,
    prompt: str,
    negative_prompt: str | None = None,
) -> tuple[str, str]:
    """Compose ``(prompt, negative_prompt)`` with the style's suffixes.

    The user's prompt text always leads; the style language follows (models
    weight earlier tokens more, so the subject stays the user's).
    """
    composed = f"{prompt.rstrip()}, {template.prompt_suffix}" if prompt.strip() else template.prompt_suffix
    negative_parts = [part for part in (negative_prompt or "", template.negative_prompt_suffix) if part]
    return composed, ", ".join(negative_parts)
