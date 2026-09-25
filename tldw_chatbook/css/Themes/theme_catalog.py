"""Single owner of the theme catalog and of switching themes (TASK-32948).

Pure data plus two small app-touching functions (``use_theme`` /
``revert_theme``); the picker, the palette and Appearance all read and
write through here so they can never disagree about which theme is
active or which one loads at launch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
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
    # Set for a saved file that can't be read (spec §4): a short, path-free
    # reason. The entry is listed so the file can be deleted.
    error: str | None = None

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
    unreadable: Mapping[str, str] | None = None,
) -> list[ThemeEntry]:
    """List every registered theme once, grouped yours → shipped → textual.

    ``unreadable`` maps each unreadable saved file's stem to its error; each
    becomes a grey "yours" entry. A stem that is already a listed id is
    skipped, since option ids must stay unique.
    """
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
    listed = {entry.id for entry in entries}
    entries.extend(
        ThemeEntry(
            id=stem,
            display_name=f"{display_name(stem)} (unreadable)",
            origin="yours",
            dark=True,
            colours=tuple((key, "#808080") for key in BASE_KEYS),
            is_active=False,
            is_launch_default=False,
            error=error,
        )
        for stem, error in (unreadable or {}).items()
        if stem not in listed
    )
    entries.sort(key=lambda e: (_ORIGIN_ORDER[e.origin], e.display_name.casefold()))
    return entries


@dataclass(frozen=True)
class ThemeChange:
    previous_active: str
    previous_launch_default: str
    persisted: bool
    # Whether the in-process config caches were refreshed after a persisted
    # write (ConfigMutationResult.caches_reloaded). True by default: a Try
    # (persist=False) never attempts a reload, so there is nothing to fail.
    caches_reloaded: bool = True

    def merge(self, later: ThemeChange) -> ThemeChange:
        """Chain a later change after this one; Revert goes back to the first.

        ``persisted`` is OR'd: if either leg actually wrote to disk, the
        chain owns a persisted state that Revert must undo. ``caches_reloaded``
        is AND'd the other way -- a failed reload in either leg must still
        surface, even if a later leg's reload happened to succeed.
        """
        return ThemeChange(
            self.previous_active,
            self.previous_launch_default,
            self.persisted or later.persisted,
            self.caches_reloaded and later.caches_reloaded,
        )


def current_launch_default() -> str:
    from ...config import get_cli_setting

    return str(get_cli_setting("general", "default_theme", "textual-dark"))


def _apply_config_mutation(mutation: dict) -> Any:
    from ...config import apply_settings_mutation_to_cli_config

    return apply_settings_mutation_to_cli_config(mutation)


def _persist_launch_default(app: Any, name: str) -> tuple[bool, bool]:
    """Write the launch default; return ``(file_replaced, caches_reloaded)``."""
    result = _apply_config_mutation({"general": {"default_theme": name}})
    file_replaced = bool(getattr(result, "file_replaced", False))
    caches_reloaded = bool(getattr(result, "caches_reloaded", False))
    if not file_replaced:
        return False, caches_reloaded
    # The in-memory copy Settings reads (was handle_theme_launch_default_changed).
    config = getattr(app, "app_config", None)
    if isinstance(config, dict):
        general = dict(config.get("general", {}))
        general["default_theme"] = name
        config["general"] = general
    return True, caches_reloaded


def persist_launch_default(app: Any, name: str) -> tuple[bool, bool]:
    """Make ``name`` the launch default without touching the running theme.

    Spec §7 step 4: renaming a non-active launch default only rewrites config.
    Returns ``(file_replaced, caches_reloaded)``.
    """
    return _persist_launch_default(app, name)


def retarget_pending_revert(app: Any, old: str, new: str | None) -> None:
    """Keep the session's pending Revert off a renamed (``new``) or deleted
    (``new is None``) theme, so Revert never targets a missing theme."""
    change: ThemeChange | None = getattr(app, "theme_revert_change", None)
    if change is None:
        return
    names = (old, f"custom_{old}")
    if change.previous_active not in names and change.previous_launch_default not in names:
        return
    if new is None:
        app.theme_revert_change = None
        return
    app.theme_revert_change = replace(
        change,
        previous_active=new if change.previous_active in names else change.previous_active,
        previous_launch_default=(
            new if change.previous_launch_default in names else change.previous_launch_default
        ),
    )


def use_theme(app: Any, name: str, *, persist: bool) -> ThemeChange:
    """Switch the app theme; with ``persist`` also make it the launch default.

    Raises textual.app.InvalidThemeError (before any write) for an
    unregistered name.
    """
    previous_active = str(app.theme)
    previous_launch = current_launch_default()
    app.theme = name
    if persist:
        persisted, caches_reloaded = _persist_launch_default(app, name)
    else:
        persisted, caches_reloaded = False, True
    return ThemeChange(previous_active, previous_launch, persisted, caches_reloaded)


def use_theme_toast(name: str, change: ThemeChange) -> tuple[str, str]:
    """Toast text + severity for a persisted (Use, not Try) theme switch.

    Fix round 1 (TASK-32948 Task 5): the palette (``ThemeProvider.switch_
    theme``) and the picker (``ThemePicker._switch``) both apply a
    persisted switch through ``use_theme`` and must show the identical
    toast for it (spec §4: one code path, one toast) -- including the
    cache-refresh-failed warning, which the picker was missing. Callers
    keep their own separate wording for a Try (``persist=False``); this
    only covers the Use branch both of them share.
    """
    if not change.persisted:
        return (
            f"{display_name(name)} applied; the launch default was not saved",
            "warning",
        )
    message = f"{display_name(name)} is now your theme (was: {display_name(change.previous_active)})"
    if change.caches_reloaded:
        return message, "information"
    return (
        f"{message}; configuration refresh failed — reopen Settings to refresh",
        "warning",
    )


def revert_theme(app: Any, change: ThemeChange) -> bool:
    """Undo ``change``; False when the launch default could not be restored."""
    app.theme = change.previous_active
    if change.persisted:
        persisted, _caches_reloaded = _persist_launch_default(app, change.previous_launch_default)
        return persisted
    return True


def user_theme_names(directory: Path) -> set[str]:
    """Stems of the saved theme files (the editor saves ``<name>.toml``)."""
    try:
        return {path.stem for path in directory.glob("*.toml")}
    except OSError:
        return set()
