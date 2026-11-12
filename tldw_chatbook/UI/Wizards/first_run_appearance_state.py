"""Pure facts and commits for the first-run Appearance step (TASK-34100.13).

coverage-15 / fulltrack-09 / fulltrack-13: the step spent half its space on
a 78-card splash gallery while the splash could not be turned off and motion
could not be reduced, and themes were listed by raw id. It now offers
"Startup animation: Off / Short (2 s) / Full", "Reduce motion" and "Plain
ASCII status marks", names the current splash card read-only, and labels
themes by display name with a tone tag. Every write is a delta: an untouched
control writes nothing, except the ASCII box's terminal preset on a first
run (see ``ascii_checkbox_value``).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from tldw_chatbook.Constants import DEFAULT_SPLASH_DURATION_SECONDS

STARTUP_OFF = "off"
STARTUP_SHORT = "short"
STARTUP_FULL = "full"
STARTUP_CHOICES = (STARTUP_OFF, STARTUP_SHORT, STARTUP_FULL)

#: "Short" plays the splash for this long (any key still skips it).
SHORT_SPLASH_SECONDS = 2.0

SPLASH_SETTINGS_PATH = "Settings ▸ Splash Screen"


def _seconds(value: float) -> str:
    return f"{value:g} s"


def startup_label(choice: str, splash: Mapping[str, Any] | None = None) -> str:
    """The radio label for one startup-animation choice.

    Args:
        choice: One of ``STARTUP_CHOICES``.
        splash: The saved ``[splash_screen]`` table. A custom length it holds
            (``custom_splash_seconds``) is named on the choice it selects,
            "Full (15 s, as saved)", since that choice keeps it.

    Returns:
        The label.
    """
    if choice == STARTUP_OFF:
        return "Off"
    word = "Short" if choice == STARTUP_SHORT else "Full"
    custom = custom_splash_seconds(splash or {})
    if custom is not None and startup_choice(splash or {}) == choice:
        return f"{word} ({_seconds(custom)}, as saved)"
    return f"{word} ({_seconds(_choice_seconds(choice))})"


def _choice_seconds(choice: str) -> float:
    if choice == STARTUP_SHORT:
        return SHORT_SPLASH_SECONDS
    return float(DEFAULT_SPLASH_DURATION_SECONDS)


def _flag(value: Any, default: bool) -> bool:
    from tldw_chatbook.config import coerce_bool_setting

    return coerce_bool_setting(value, default)


def _saved_seconds(splash: Mapping[str, Any]) -> float:
    """The saved splash length; the default when unset or unreadable."""
    try:
        duration = float(splash.get("duration", DEFAULT_SPLASH_DURATION_SECONDS))
    except (TypeError, ValueError):
        return float(DEFAULT_SPLASH_DURATION_SECONDS)
    return duration if math.isfinite(duration) else float(DEFAULT_SPLASH_DURATION_SECONDS)


def startup_choice(splash: Mapping[str, Any]) -> str:
    """Which choice the saved ``[splash_screen]`` table amounts to."""
    if not _flag(splash.get("enabled", True), True):
        return STARTUP_OFF
    duration = _saved_seconds(splash)
    return STARTUP_SHORT if duration <= SHORT_SPLASH_SECONDS else STARTUP_FULL


def custom_splash_seconds(splash: Mapping[str, Any]) -> float | None:
    """A saved splash length that is neither Short's nor the default, else None.

    Pre-merge review (2026-11-12, TASK-34100.13): a length set in Settings ▸
    Splash Screen (4 s, 15 s, 1 s) read as one of setup's two -- "full splash
    (7 s)" or "short splash (2 s)". Callers name these seconds instead.

    Args:
        splash: The saved ``[splash_screen]`` table.

    Returns:
        The seconds, or None when the splash is off or the length is 2 s or
        the default.
    """
    if startup_choice(splash) == STARTUP_OFF:
        return None
    duration = _saved_seconds(splash)
    if duration in (SHORT_SPLASH_SECONDS, float(DEFAULT_SPLASH_DURATION_SECONDS)):
        return None
    return duration


def build_startup_commit(*, choice: str, current: str) -> dict[str, dict[str, Any]]:
    """The ``[splash_screen]`` delta for a startup choice; ``{}`` when unchanged."""
    if choice == current or choice not in STARTUP_CHOICES:
        return {}
    if choice == STARTUP_OFF:
        return {"splash_screen": {"enabled": False}}
    duration = (
        SHORT_SPLASH_SECONDS
        if choice == STARTUP_SHORT
        else float(DEFAULT_SPLASH_DURATION_SECONDS)
    )
    return {"splash_screen": {"enabled": True, "duration": duration}}


def read_comfort(appearance: Mapping[str, Any]) -> tuple[bool, bool]:
    """``(reduce_motion, ascii_glyphs)`` from an ``[appearance]`` table."""
    return (
        _flag(appearance.get("reduce_motion", False), False),
        _flag(appearance.get("ascii_glyphs", False), False),
    )


def ascii_preset_from_environment(environ: Mapping[str, str]) -> bool:
    """Whether the terminal suggests plain ASCII marks (NO_COLOR, or the Linux console)."""
    return bool(environ.get("NO_COLOR")) or environ.get("TERM", "") == "linux"


def ascii_checkbox_value(
    appearance: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    rerun: bool = False,
) -> bool:
    """The ASCII box's starting value: the saved choice, else the terminal preset.

    Review F2 (TASK-34100.13): the preset stands in only for a choice never
    made. On a re-run a saved ``ascii_glyphs = false`` stays unticked under
    NO_COLOR, so Next on an untouched step writes nothing. On a first run the
    value in config.toml is the shipped template's ``ascii_glyphs = false``
    (every new profile has it), not a choice, so the preset is the step's
    proposal there and Next saves it like any other change.

    Args:
        appearance: The full-TOML ``[appearance]`` table.
        environ: The process environment.
        rerun: Whether setup has run before (Settings or the palette).

    Returns:
        Whether the box starts ticked.
    """
    saved = read_comfort(appearance)[1]
    if rerun and "ascii_glyphs" in appearance:
        return saved
    return saved or ascii_preset_from_environment(environ)


def build_comfort_commit(
    *,
    reduce_motion: bool,
    ascii_glyphs: bool,
    current_reduce_motion: bool,
    current_ascii_glyphs: bool,
) -> dict[str, dict[str, Any]]:
    """The ``[appearance]`` delta for the two comfort checkboxes."""
    changed: dict[str, Any] = {}
    if bool(reduce_motion) != bool(current_reduce_motion):
        changed["reduce_motion"] = bool(reduce_motion)
    if bool(ascii_glyphs) != bool(current_ascii_glyphs):
        changed["ascii_glyphs"] = bool(ascii_glyphs)
    return {"appearance": changed} if changed else {}


_THEME_NAMES: Mapping[str, str] = {
    "textual-dark": "Default dark",
    "textual-light": "Default light",
}


def theme_display_label(theme_id: str, *, dark: bool | None) -> str:
    """A theme's display name with a tone tag ("Nord · dark").

    Args:
        theme_id: The theme's registered name.
        dark: Its tone, or None when unknown (the tag is then left off).

    Returns:
        The label; the raw id always rides on the row separately.
    """
    fixed = _THEME_NAMES.get(theme_id)
    if fixed:
        return fixed
    words = theme_id.replace("_", "-").split("-")
    name = " ".join(word[:1].upper() + word[1:] for word in words if word)
    if dark is None:
        return name or theme_id
    return f"{name or theme_id} · {'dark' if dark else 'light'}"


def theme_tone(theme_id: str, themes: Mapping[str, Any] | None = None) -> bool | None:
    """A theme's dark flag from ``themes`` (an app's), else Textual's built-ins."""
    theme = (themes or {}).get(theme_id)
    if theme is None:
        try:
            from textual.theme import BUILTIN_THEMES

            theme = BUILTIN_THEMES.get(theme_id)
        except Exception:  # noqa: BLE001 -- a label must never break setup
            theme = None
    dark = getattr(theme, "dark", None)
    return dark if isinstance(dark, bool) else None


def card_display_name(card_selection: str) -> str:
    """The current splash card in words ("Surprise me" for random)."""
    card = (card_selection or "").strip()
    if not card or card in {"random", "sequential"}:
        return "Surprise me (a random card each launch)"
    return card.replace("_", " ").strip().title()


def splash_card_line(card_selection: str) -> str:
    """The one read-only line that replaces the splash-card gallery."""
    return (
        f"Splash card: {card_display_name(card_selection)} — preview and pick "
        f"one in {SPLASH_SETTINGS_PATH}."
    )


def startup_summary(
    splash: Mapping[str, Any], appearance: Mapping[str, Any]
) -> tuple[bool, str]:
    """``(changed_from_defaults, detail)`` for the Summary's startup row."""
    choice = startup_choice(splash)
    custom = custom_splash_seconds(splash)
    reduce_motion, ascii_glyphs = read_comfort(appearance)
    if choice == STARTUP_OFF:
        parts = ["no splash"]
    else:
        word = "short" if choice == STARTUP_SHORT else "full"
        seconds = _choice_seconds(choice) if custom is None else custom
        parts = [f"{word} splash ({_seconds(seconds)})"]
    if reduce_motion:
        parts.append("motion reduced")
    if ascii_glyphs:
        parts.append("plain ASCII marks")
    changed = (
        choice != STARTUP_FULL or custom is not None or reduce_motion or ascii_glyphs
    )
    return changed, " · ".join(parts)
