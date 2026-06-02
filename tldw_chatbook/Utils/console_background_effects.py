"""Console background effect settings and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


CONSOLE_BACKGROUND_EFFECTS = frozenset({"none", "snow", "rain", "matrix"})
CONSOLE_BACKGROUND_SCOPES = frozenset({"transcript", "workbench"})
CONSOLE_BACKGROUND_INTENSITIES = frozenset({"low", "medium", "high"})
DEFAULT_CONSOLE_BACKGROUND_FPS = 6
MIN_CONSOLE_BACKGROUND_FPS = 1
MAX_CONSOLE_BACKGROUND_FPS = 12


@dataclass(frozen=True)
class ConsoleBackgroundEffectSettings:
    """Normalized Console background effect configuration."""

    enabled: bool = False
    effect: str = "none"
    scope: str = "transcript"
    intensity: str = "low"
    fps: int = DEFAULT_CONSOLE_BACKGROUND_FPS

    @property
    def active(self) -> bool:
        """Return whether an effect should render."""
        return self.enabled and self.effect != "none"

    def to_config(self) -> dict[str, object]:
        """Serialize settings to the app config dictionary shape."""
        return {
            "enabled": self.enabled,
            "effect": self.effect,
            "scope": self.scope,
            "intensity": self.intensity,
            "fps": self.fps,
        }


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _coerce_choice(value: object, allowed: frozenset[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else default


def _coerce_fps(value: object) -> int:
    if isinstance(value, bool):
        return DEFAULT_CONSOLE_BACKGROUND_FPS
    try:
        fps = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CONSOLE_BACKGROUND_FPS
    return max(MIN_CONSOLE_BACKGROUND_FPS, min(MAX_CONSOLE_BACKGROUND_FPS, fps))


def normalize_console_background_effects(
    values: Mapping[str, object] | None,
) -> ConsoleBackgroundEffectSettings:
    """Normalize raw config values to safe Console background effect settings."""
    raw = values if isinstance(values, Mapping) else {}
    return ConsoleBackgroundEffectSettings(
        enabled=_coerce_bool(raw.get("enabled"), False),
        effect=_coerce_choice(raw.get("effect"), CONSOLE_BACKGROUND_EFFECTS, "none"),
        scope=_coerce_choice(raw.get("scope"), CONSOLE_BACKGROUND_SCOPES, "transcript"),
        intensity=_coerce_choice(
            raw.get("intensity"),
            CONSOLE_BACKGROUND_INTENSITIES,
            "low",
        ),
        fps=_coerce_fps(raw.get("fps")),
    )
