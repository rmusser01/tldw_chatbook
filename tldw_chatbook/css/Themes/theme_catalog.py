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
from typing import Literal

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
        try:
            pairs.append((key, Color.parse(str(raw)).hex.upper() if raw else "#808080"))
        except Exception:  # noqa: BLE001
            pairs.append((key, "#808080"))
    return tuple(pairs)


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
