"""Deterministic prompt refinement helpers for image generation."""

from __future__ import annotations

from typing import Any

DEFAULT_PROMPT_REFINEMENT_MODE = "auto"
DEFAULT_QUALITY_SUFFIX = "high detail, coherent composition, natural lighting, sharp focus"

_KNOWN_MODES = {"off", "auto", "basic"}
_QUALITY_CUES = (
    "high detail",
    "highly detailed",
    "high quality",
    "ultra detailed",
    "photorealistic",
    "cinematic",
    "composition",
    "lighting",
    "sharp focus",
    "8k",
)


def normalize_prompt_refinement_mode(value: Any, *, default: str = DEFAULT_PROMPT_REFINEMENT_MODE) -> str:
    """Normalize prompt refinement mode from bool/string input."""
    normalized_default = str(default or DEFAULT_PROMPT_REFINEMENT_MODE).strip().lower()
    if normalized_default not in _KNOWN_MODES:
        normalized_default = DEFAULT_PROMPT_REFINEMENT_MODE

    if isinstance(value, bool):
        return "basic" if value else "off"
    if value is None:
        return normalized_default

    raw = str(value).strip().lower()
    if not raw:
        return normalized_default
    if raw in {"off", "none", "false", "0", "no"}:
        return "off"
    if raw in {"on", "true", "1", "yes", "basic"}:
        return "basic"
    if raw in {"auto"}:
        return "auto"
    return normalized_default


def refine_image_prompt(
    prompt: str,
    *,
    mode: Any = DEFAULT_PROMPT_REFINEMENT_MODE,
    quality_suffix: str = DEFAULT_QUALITY_SUFFIX,
    max_length: int | None = None,
) -> str:
    """Apply deterministic quality refinement to sparse prompts."""
    normalized_prompt = _normalize_spaces(prompt)
    if not normalized_prompt:
        return ""

    mode_name = normalize_prompt_refinement_mode(mode)
    if mode_name == "off":
        return normalized_prompt

    should_enrich = mode_name == "basic" or _needs_quality_guidance(normalized_prompt)
    if not should_enrich:
        return normalized_prompt

    suffix = _normalize_spaces(quality_suffix)
    if not suffix:
        return normalized_prompt
    if suffix.lower() in normalized_prompt.lower():
        return normalized_prompt

    candidate = f"{normalized_prompt}, {suffix}"
    if isinstance(max_length, int) and max_length > 0 and len(candidate) > max_length:
        return normalized_prompt
    return candidate


def _needs_quality_guidance(prompt: str) -> bool:
    lowered = prompt.lower()
    if any(cue in lowered for cue in _QUALITY_CUES):
        return False
    word_count = len(prompt.split())
    if word_count >= 14:
        return False
    return True


def _normalize_spaces(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()

