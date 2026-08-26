"""Appearance defaults exposed by the Settings hub."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat.console_roleplay_identity import (
    DEFAULT_CONSOLE_TRANSCRIPT_STYLE,
    ConsoleTranscriptStyle,
    normalize_console_transcript_style,
)
from tldw_chatbook.Utils.adaptive_reader_state import (
    ITEMS_MAX_WIDTH,
    ITEMS_MIN_WIDTH,
    ITEMS_TARGET_WIDTH,
    normalize_adaptive_reader_preferences,
)
from tldw_chatbook.Library.library_rail_width import (
    LIBRARY_CUSTOM_MAX_WIDTH,
    LIBRARY_MIN_WIDTH,
    LIBRARY_REFERENCE_WIDTH,
)

from .settings_config_models import SettingsValidationResult


DEFAULT_THEME = "textual-dark"
DEFAULT_PALETTE_THEME_LIMIT = 1
DEFAULT_FONT_SIZE = 12
DEFAULT_DENSITY = "normal"
DEFAULT_ANIMATIONS_ENABLED = True
DEFAULT_SMOOTH_SCROLLING = True
# TASK-2154.10 (AC-04): static-frame rendering for splash + setup backdrop.
DEFAULT_REDUCE_MOTION = False
# TASK-2154.19 (AC-01): ASCII-safe status markers for narrow-font terminals.
DEFAULT_ASCII_GLYPHS = False
DEFAULT_CONSOLE_TRANSCRIPT_STYLE_VALUE = DEFAULT_CONSOLE_TRANSCRIPT_STYLE.value
SUPPORTED_DENSITIES = frozenset({"compact", "normal", "comfortable"})
SUPPORTED_CONSOLE_TRANSCRIPT_STYLES = frozenset(
    style.value for style in ConsoleTranscriptStyle
)
MIN_PALETTE_THEME_LIMIT = 0
MAX_PALETTE_THEME_LIMIT = 100
MIN_FONT_SIZE = 6
MAX_FONT_SIZE = 32
MAX_THEME_NAME_LENGTH = 128


@dataclass(frozen=True)
class SettingsAppearanceDefaults:
    """Editable Appearance defaults exposed in Settings."""

    default_theme: str = DEFAULT_THEME
    palette_theme_limit: int = DEFAULT_PALETTE_THEME_LIMIT
    font_size: int = DEFAULT_FONT_SIZE
    density: str = DEFAULT_DENSITY
    animations_enabled: bool = DEFAULT_ANIMATIONS_ENABLED
    smooth_scrolling: bool = DEFAULT_SMOOTH_SCROLLING
    reduce_motion: bool = DEFAULT_REDUCE_MOTION
    ascii_glyphs: bool = DEFAULT_ASCII_GLYPHS
    console_transcript_style: str = DEFAULT_CONSOLE_TRANSCRIPT_STYLE_VALUE
    library_reader_library_open: bool = True
    library_reader_custom_widths_enabled: bool = False
    library_reader_library_width: int = LIBRARY_REFERENCE_WIDTH
    library_media_items_open: bool = True
    library_media_items_width: int = ITEMS_TARGET_WIDTH
    library_conversations_items_open: bool = True
    library_conversations_items_width: int = ITEMS_TARGET_WIDTH
    library_notes_items_open: bool = True
    library_notes_items_width: int = ITEMS_TARGET_WIDTH
    library_prompts_items_open: bool = True
    library_prompts_items_width: int = ITEMS_TARGET_WIDTH
    library_skills_items_open: bool = True
    library_skills_items_width: int = ITEMS_TARGET_WIDTH


def _mapping_child(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a child mapping or an empty mapping when absent."""
    child = parent.get(key, {})
    return child if isinstance(child, Mapping) else {}


def _coerce_bool(value: Any, default: bool) -> bool:
    """Coerce common config boolean values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    """Coerce integral config values with a safe fallback."""
    if isinstance(value, bool):
        return default
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if not parsed.is_integer():
        return default
    return int(parsed)


def _strict_int(value: Any) -> int | None:
    """Return an integer only when the value is unambiguous."""
    if isinstance(value, bool):
        return None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not parsed.is_integer():
        return None
    return int(parsed)


def _strict_bool(value: Any) -> bool | None:
    """Return a boolean only when the value is already boolean."""
    return value if isinstance(value, bool) else None


def _normalise_theme(value: Any) -> str:
    """Return a non-empty theme name or the default theme."""
    theme = str(value or "").strip()
    if not theme:
        return DEFAULT_THEME
    return theme[:MAX_THEME_NAME_LENGTH]


def _normalise_density(value: Any) -> str:
    """Return a supported density or the default density."""
    density = str(value or "").strip().lower()
    return density if density in SUPPORTED_DENSITIES else DEFAULT_DENSITY


def load_appearance_defaults(
    app_config: Mapping[str, Any],
) -> SettingsAppearanceDefaults:
    """Load Settings-owned Appearance defaults from app configuration.

    Args:
        app_config: Application configuration mapping to read from.

    Returns:
        Coerced Appearance defaults with safe fallbacks for missing or malformed
        values.
    """
    general = _mapping_child(app_config, "general")
    web_server = _mapping_child(app_config, "web_server")
    appearance = _mapping_child(app_config, "appearance")
    library = _mapping_child(app_config, "library")
    legacy_media_reader = _mapping_child(library, "media_reader")
    raw_reader = _mapping_child(library, "reader")
    shared_raw = {
        key: raw_reader.get(key, legacy_media_reader.get(key))
        for key in ("library_open", "custom_widths_enabled", "library_width")
    }
    shared_reader = normalize_adaptive_reader_preferences(shared_raw)
    shared_width = normalize_adaptive_reader_preferences(
        {**shared_raw, "custom_widths_enabled": True}
    ).library_width

    def destination_reader(section: str):
        raw_destination = _mapping_child(library, section)
        return normalize_adaptive_reader_preferences(
            {
                "custom_widths_enabled": True,
                "items_open": raw_destination.get("items_open"),
                "items_width": raw_destination.get("items_width"),
            }
        )

    destination_readers = {
        name: destination_reader(f"{name}_reader")
        for name in ("media", "conversations", "notes", "prompts", "skills")
    }

    return SettingsAppearanceDefaults(
        default_theme=_normalise_theme(general.get("default_theme", DEFAULT_THEME)),
        palette_theme_limit=_coerce_int(
            general.get("palette_theme_limit", DEFAULT_PALETTE_THEME_LIMIT),
            DEFAULT_PALETTE_THEME_LIMIT,
        ),
        font_size=_coerce_int(
            web_server.get("font_size", DEFAULT_FONT_SIZE),
            DEFAULT_FONT_SIZE,
        ),
        density=_normalise_density(appearance.get("density", DEFAULT_DENSITY)),
        animations_enabled=_coerce_bool(
            appearance.get("animations_enabled", DEFAULT_ANIMATIONS_ENABLED),
            DEFAULT_ANIMATIONS_ENABLED,
        ),
        smooth_scrolling=_coerce_bool(
            appearance.get("smooth_scrolling", DEFAULT_SMOOTH_SCROLLING),
            DEFAULT_SMOOTH_SCROLLING,
        ),
        reduce_motion=_coerce_bool(
            appearance.get("reduce_motion", DEFAULT_REDUCE_MOTION),
            DEFAULT_REDUCE_MOTION,
        ),
        ascii_glyphs=_coerce_bool(
            appearance.get("ascii_glyphs", DEFAULT_ASCII_GLYPHS),
            DEFAULT_ASCII_GLYPHS,
        ),
        console_transcript_style=normalize_console_transcript_style(
            appearance.get(
                "console_transcript_style",
                DEFAULT_CONSOLE_TRANSCRIPT_STYLE_VALUE,
            )
        ).value,
        library_reader_library_open=shared_reader.library_open,
        library_reader_custom_widths_enabled=(
            shared_reader.custom_widths_enabled
        ),
        library_reader_library_width=shared_width,
        library_media_items_open=destination_readers["media"].items_open,
        library_media_items_width=destination_readers["media"].items_width,
        library_conversations_items_open=(
            destination_readers["conversations"].items_open
        ),
        library_conversations_items_width=(
            destination_readers["conversations"].items_width
        ),
        library_notes_items_open=destination_readers["notes"].items_open,
        library_notes_items_width=destination_readers["notes"].items_width,
        library_prompts_items_open=destination_readers["prompts"].items_open,
        library_prompts_items_width=destination_readers["prompts"].items_width,
        library_skills_items_open=destination_readers["skills"].items_open,
        library_skills_items_width=destination_readers["skills"].items_width,
    )


def validate_appearance_defaults(
    values: SettingsAppearanceDefaults,
) -> SettingsValidationResult:
    """Validate editable Appearance defaults before persistence.

    Args:
        values: Appearance defaults to validate.

    Returns:
        Validation state and user-facing recovery copy.
    """
    theme = str(values.default_theme or "").strip()
    if not theme:
        return SettingsValidationResult(False, "Theme is required.")
    if len(theme) > MAX_THEME_NAME_LENGTH:
        return SettingsValidationResult(
            False,
            f"Theme must be {MAX_THEME_NAME_LENGTH} characters or fewer.",
        )
    palette_theme_limit = _strict_int(values.palette_theme_limit)
    if (
        palette_theme_limit is None
        or not MIN_PALETTE_THEME_LIMIT <= palette_theme_limit <= MAX_PALETTE_THEME_LIMIT
    ):
        return SettingsValidationResult(
            False,
            "Palette theme limit must be between "
            f"{MIN_PALETTE_THEME_LIMIT} and {MAX_PALETTE_THEME_LIMIT}.",
        )
    font_size = _strict_int(values.font_size)
    if font_size is None or not MIN_FONT_SIZE <= font_size <= MAX_FONT_SIZE:
        return SettingsValidationResult(
            False,
            f"Font size must be between {MIN_FONT_SIZE} and {MAX_FONT_SIZE}.",
        )
    if str(values.density).strip().lower() not in SUPPORTED_DENSITIES:
        return SettingsValidationResult(
            False,
            "Density must be compact, normal, or comfortable.",
        )
    if _strict_bool(values.animations_enabled) is None:
        return SettingsValidationResult(
            False,
            "Animations must be enabled or disabled.",
        )
    if _strict_bool(values.smooth_scrolling) is None:
        return SettingsValidationResult(
            False,
            "Smooth scrolling must be enabled or disabled.",
        )
    if _strict_bool(values.reduce_motion) is None:
        return SettingsValidationResult(
            False,
            "Reduce motion must be enabled or disabled.",
        )
    if _strict_bool(values.ascii_glyphs) is None:
        return SettingsValidationResult(
            False,
            "ASCII glyphs must be enabled or disabled.",
        )
    if str(values.console_transcript_style) not in SUPPORTED_CONSOLE_TRANSCRIPT_STYLES:
        return SettingsValidationResult(
            False,
            "Transcript style must be neutral, role accents, or immersive RP.",
        )
    if _strict_bool(values.library_reader_library_open) is None:
        return SettingsValidationResult(
            False, "Library pane preference must be open or collapsed."
        )
    if _strict_bool(values.library_reader_custom_widths_enabled) is None:
        return SettingsValidationResult(
            False, "Custom widths must be enabled or disabled."
        )
    library_width = _strict_int(values.library_reader_library_width)
    if (
        library_width is None
        or not LIBRARY_MIN_WIDTH <= library_width <= LIBRARY_CUSTOM_MAX_WIDTH
    ):
        return SettingsValidationResult(
            False,
            "Library width must be between "
            f"{LIBRARY_MIN_WIDTH} and {LIBRARY_CUSTOM_MAX_WIDTH}.",
        )
    for label, open_value, width_value in (
        (
            "Items",
            values.library_media_items_open,
            values.library_media_items_width,
        ),
        (
            "Conversations Items",
            values.library_conversations_items_open,
            values.library_conversations_items_width,
        ),
        (
            "Notes Items",
            values.library_notes_items_open,
            values.library_notes_items_width,
        ),
        (
            "Prompts Items",
            values.library_prompts_items_open,
            values.library_prompts_items_width,
        ),
        (
            "Skills Items",
            values.library_skills_items_open,
            values.library_skills_items_width,
        ),
    ):
        if _strict_bool(open_value) is None:
            return SettingsValidationResult(
                False, f"{label} pane preference must be open or collapsed."
            )
        items_width = _strict_int(width_value)
        if (
            items_width is None
            or not ITEMS_MIN_WIDTH <= items_width <= ITEMS_MAX_WIDTH
        ):
            return SettingsValidationResult(
                False,
                f"{label} width must be between {ITEMS_MIN_WIDTH} and {ITEMS_MAX_WIDTH}.",
            )
    return SettingsValidationResult(True, "Appearance defaults are valid.")


def build_appearance_save_sections(
    app_config: Mapping[str, Any],
    values: SettingsAppearanceDefaults,
) -> dict[str, dict[str, Any]]:
    """Build config sections needed to persist Appearance defaults.

    Args:
        app_config: Existing application configuration mapping.
        values: Validated Appearance defaults to persist.

    Returns:
        A mapping of config section names to deep-merged section values.
    """
    general = dict(deepcopy(_mapping_child(app_config, "general")))
    web_server = dict(deepcopy(_mapping_child(app_config, "web_server")))
    appearance = dict(deepcopy(_mapping_child(app_config, "appearance")))
    library = dict(deepcopy(_mapping_child(app_config, "library")))
    reader = dict(deepcopy(_mapping_child(library, "reader")))

    general.update(
        {
            "default_theme": str(values.default_theme).strip(),
            "palette_theme_limit": int(values.palette_theme_limit),
        }
    )
    web_server.update({"font_size": int(values.font_size)})
    appearance.update(
        {
            "density": str(values.density).strip().lower(),
            "animations_enabled": bool(values.animations_enabled),
            "smooth_scrolling": bool(values.smooth_scrolling),
            "reduce_motion": bool(values.reduce_motion),
            "ascii_glyphs": bool(values.ascii_glyphs),
            "console_transcript_style": str(values.console_transcript_style),
        }
    )
    reader.update(
        {
            "library_open": bool(values.library_reader_library_open),
            "custom_widths_enabled": bool(
                values.library_reader_custom_widths_enabled
            ),
            "library_width": int(values.library_reader_library_width),
        }
    )
    library["reader"] = reader
    for destination in ("media", "conversations", "notes", "prompts", "skills"):
        section_name = f"{destination}_reader"
        destination_reader = dict(
            deepcopy(_mapping_child(library, section_name))
        )
        destination_reader.update(
            {
                "items_open": bool(
                    getattr(values, f"library_{destination}_items_open")
                ),
                "items_width": int(
                    getattr(values, f"library_{destination}_items_width")
                ),
            }
        )
        library[section_name] = destination_reader

    return {
        "general": general,
        "web_server": web_server,
        "appearance": appearance,
        "library": library,
    }
