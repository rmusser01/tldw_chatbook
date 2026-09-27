"""Single owner of the theme catalog and of switching themes (TASK-32948).

Pure data plus two small app-touching functions (``use_theme`` /
``revert_theme``); the picker, the palette and Appearance all read and
write through here so they can never disagree about which theme is
active or which one loads at launch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from textual.color import Color
from textual.theme import BUILTIN_THEMES, Theme

from .themes import ALL_THEMES

Origin = Literal["yours", "shipped", "textual"]
STRIP_KEYS = ("background", "surface", "primary", "secondary", "accent", "success", "error")
BASE_KEYS = (
    "primary", "secondary", "accent", "background", "surface",
    "panel", "foreground", "success", "warning", "error",
)
_ORIGIN_ORDER: dict[str, int] = {"yours": 0, "shipped": 1, "textual": 2}
_SHIPPED_NAMES = frozenset(t.name for t in ALL_THEMES if getattr(t, "name", None))


@dataclass(frozen=True)
class ThemeEntry:
    id: str
    display_name: str
    origin: Origin
    dark: bool
    colours: tuple[tuple[str, str], ...]
    is_active: bool
    is_launch_default: bool
    overrides: Literal["shipped", "textual"] | None = None

    @property
    def strip(self) -> tuple[str, ...]:
        palette = dict(self.colours)
        return tuple(palette[key] for key in STRIP_KEYS)


def display_name(theme_id: str) -> str:
    return theme_id.replace("_", " ").replace("-", " ").title()


def is_catalog_theme(name: str) -> bool:
    """True for a shipped or Textual built-in theme name."""
    return name in _SHIPPED_NAMES or name in BUILTIN_THEMES


def _colours(theme: Theme) -> tuple[tuple[str, str], ...]:
    # Same resolution as the editor (TASK-31255): explicit colours byte-exact,
    # unset ones from the generated colour system.
    try:
        generated = theme.to_color_system().generate()
    except Exception:  # noqa: BLE001 - a malformed theme must not break the list
        generated = {}
    pairs: list[tuple[str, str]] = []
    for key in BASE_KEYS:
        raw = getattr(theme, key, None) or generated.get(key)
        pairs.append((key, _colour_hex(raw)))
    return tuple(pairs)


def _colour_hex(raw: Any) -> str:
    """Resolve one theme colour to an uppercase 6-digit ``#RRGGBB``.

    Every consumer (the picker's colour strips, ``ThemePreview.paint``, ...)
    needs a plain RGB hex -- never one with an alpha suffix (``Color.hex``
    returns 8 digits, e.g. ``#FF33AACC``, once alpha < 1) and never an ANSI
    colour name (``Color.hex`` returns e.g. ``"ansi_default"`` for the
    ansi-dark/ansi-light builtins, since those resolve against the terminal's
    own palette at render time and have no RGB of their own to report). Use
    ``hex6`` to always drop alpha, and fall back to neutral grey for ANSI
    colours the same way an unset colour already falls back.
    """
    if not raw:
        return "#808080"
    try:
        colour = Color.parse(str(raw))
    except Exception:  # noqa: BLE001
        return "#808080"
    if colour.ansi is not None:
        return "#808080"
    return colour.hex6.upper()


def _origin(name: str, user_names: Collection[str]) -> tuple[Origin, Literal["shipped", "textual"] | None]:
    if name in user_names:
        if name in _SHIPPED_NAMES:
            return "yours", "shipped"
        if name in BUILTIN_THEMES:
            return "yours", "textual"
        return "yours", None
    if name in BUILTIN_THEMES and name not in _SHIPPED_NAMES:
        return "textual", None
    return "shipped", None


def build_catalog(
    available: Mapping[str, Theme],
    user_names: Collection[str],
    active: str,
    launch_default: str,
) -> list[ThemeEntry]:
    """List every registered theme once, grouped yours → shipped → textual."""
    rows: list[tuple[str, Origin, Literal["shipped", "textual"] | None, Theme]] = []
    for name, theme in available.items():
        if name.startswith("custom_"):
            continue  # Apply's process-only registration (PR #2375 #8)
        origin, overrides = _origin(name, user_names)
        rows.append((name, origin, overrides, theme))
    counts = Counter(display_name(name) for name, *_ in rows)
    entries = [
        ThemeEntry(
            id=name,
            display_name=(
                f"{display_name(name)} · {name}" if counts[display_name(name)] > 1 else display_name(name)
            ),
            origin=origin,
            dark=bool(getattr(theme, "dark", True)),
            colours=_colours(theme),
            # Editor Apply runs `custom_<name>`; that is still <name> in use.
            is_active=active in (name, f"custom_{name}"),
            is_launch_default=name == launch_default,
            overrides=overrides,
        )
        for name, origin, overrides, theme in rows
    ]
    entries.sort(key=lambda e: (_ORIGIN_ORDER[e.origin], e.display_name.casefold()))
    return entries


@dataclass(frozen=True)
class ThemeChange:
    previous_active: str
    previous_launch_default: str
    persisted: bool

    def merge(self, later: ThemeChange) -> ThemeChange:
        """Chain a later change after this one; Revert goes back to the first."""
        return ThemeChange(self.previous_active, self.previous_launch_default, self.persisted or later.persisted)


def current_launch_default() -> str:
    from ...config import get_cli_setting

    return str(get_cli_setting("general", "default_theme", "textual-dark"))


def _apply_config_mutation(mutation: dict) -> Any:
    from ...config import apply_settings_mutation_to_cli_config

    return apply_settings_mutation_to_cli_config(mutation)


def _persist_launch_default(app: Any, name: str) -> bool:
    result = _apply_config_mutation({"general": {"default_theme": name}})
    if not getattr(result, "file_replaced", False):
        return False
    # The in-memory copy Settings reads (was handle_theme_launch_default_changed).
    config = getattr(app, "app_config", None)
    if isinstance(config, dict):
        general = dict(config.get("general", {}))
        general["default_theme"] = name
        config["general"] = general
    return True


def use_theme(app: Any, name: str, *, persist: bool) -> ThemeChange:
    """Switch the app theme; with ``persist`` also make it the launch default.

    Raises textual.app.InvalidThemeError (before any write) for an
    unregistered name.
    """
    previous_active = str(app.theme)
    previous_launch = current_launch_default()
    app.theme = name
    persisted = _persist_launch_default(app, name) if persist else False
    return ThemeChange(previous_active, previous_launch, persisted)


def revert_theme(app: Any, change: ThemeChange) -> bool:
    """Undo ``change``; False when the launch default could not be restored."""
    app.theme = change.previous_active
    if change.persisted:
        return _persist_launch_default(app, change.previous_launch_default)
    return True


def user_theme_names(directory: Path) -> set[str]:
    """Stems of the saved theme files (the editor saves ``<name>.toml``)."""
    try:
        return {path.stem for path in directory.glob("*.toml")}
    except OSError:
        return set()
