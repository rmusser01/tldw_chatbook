"""Appearance defaults exposed by the Settings hub."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .settings_config_models import SettingsValidationResult


DEFAULT_THEME = "textual-dark"
DEFAULT_PALETTE_THEME_LIMIT = 1
DEFAULT_FONT_SIZE = 12
DEFAULT_DENSITY = "normal"
DEFAULT_ANIMATIONS_ENABLED = True
DEFAULT_SMOOTH_SCROLLING = True
SUPPORTED_DENSITIES = frozenset({"compact", "normal", "comfortable"})
MIN_PALETTE_THEME_LIMIT = 0
MAX_PALETTE_THEME_LIMIT = 100
MIN_FONT_SIZE = 8
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


def _mapping_child(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a child mapping or an empty mapping when absent."""
    child = parent.get(key, {})
    return child if isinstance(child, Mapping) else {}


def _coerce_bool(value: Any, default: bool) -> bool:
    """Coerce common config boolean values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
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


def load_appearance_defaults(app_config: Mapping[str, Any]) -> SettingsAppearanceDefaults:
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
        }
    )

    return {
        "general": general,
        "web_server": web_server,
        "appearance": appearance,
    }
