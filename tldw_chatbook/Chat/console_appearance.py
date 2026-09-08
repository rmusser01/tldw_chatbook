"""Versioned conversation metadata for Console appearance customization.

task-31207: per-conversation icon + color, Cursor-style. Stored under one
namespaced key in ``conversations.metadata`` JSON — the same contract
``console_speech`` and ``console_roleplay_context`` follow — so it syncs with
the conversation and needs no schema migration.

Read-side sanitization is per member (a malformed icon keeps a valid color),
but a payload whose key set is not exactly ``{"icon", "color"}`` is dropped
whole, mirroring ``console_speech``'s version-skew protection: a newer shape
must not be partially interpreted by this version.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len

CONSOLE_APPEARANCE_METADATA_KEY = "console_appearance"

#: Fixed swatch palette offered by the appearance picker: (label, hex). The
#: hex values are tuned for dark rail panels; Rich approximates them on
#: 256-color terminals. TASK-31209 may extend this list.
CONSOLE_APPEARANCE_PALETTE: tuple[tuple[str, str], ...] = (
    ("Red", "#f87171"),
    ("Orange", "#fb923c"),
    ("Amber", "#fbbf24"),
    ("Green", "#4ade80"),
    ("Teal", "#2dd4bf"),
    ("Cyan", "#22d3ee"),
    ("Sky", "#38bdf8"),
    ("Blue", "#60a5fa"),
    ("Indigo", "#818cf8"),
    ("Violet", "#a78bfa"),
    ("Magenta", "#e879f9"),
    ("Pink", "#f472b6"),
    ("Gray", "#9ca3af"),
    # task-31209: the extended row — deeper/darker siblings of the base
    # hues plus fills for uncovered parts of the wheel. All still
    # truecolor-readable on dark rail panels.
    ("Rose", "#fb7185"),
    ("Ruby", "#e11d48"),
    ("Gold", "#eab308"),
    ("Lime", "#a3e635"),
    ("Emerald", "#10b981"),
    ("Deep Blue", "#3b82f6"),
    ("Purple", "#c084fc"),
    ("Fuchsia", "#d946ef"),
)

CONSOLE_APPEARANCE_COLOR_HEXES: frozenset[str] = frozenset(
    value for _, value in CONSOLE_APPEARANCE_PALETTE
)

_COLOR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}\Z")
_VARIATION_SELECTORS = ("\ufe0f", "\ufe0e")


def is_valid_console_appearance_icon(value: object) -> bool:
    """Return whether value is one renderable grapheme of cell width <= 2.

    A single code point, optionally followed by a variation selector, keeps
    common picker output like ``"❤️"`` (U+2764 U+FE0F) valid while rejecting
    multi-glyph strings, ZWJ families, and wide pairs that would break the
    row's hand-computed width budget.
    """
    if type(value) is not str or not value:
        return False
    stripped = value
    while stripped.endswith(_VARIATION_SELECTORS):
        stripped = stripped[:-1]
    if len(stripped) != 1:
        return False
    return cell_len(value) <= 2


def is_valid_console_appearance_color(value: object) -> bool:
    """Return whether value is a canonical ``#rrggbb`` hex color.

    Format-only by design: the picker restricts swatch writes to the
    palette, but reads (and the custom-hex entry, task-31209) stay
    forward-compatible with any canonical hex value.
    """
    return type(value) is str and _COLOR_PATTERN.fullmatch(value) is not None


def normalize_console_appearance_hex_input(value: object) -> str | None:
    """Normalize typed color input to canonical ``#rrggbb``, or None.

    task-31209: the picker's custom-color field accepts a leading ``#``
    optionally (``c0ffee`` == ``#c0ffee``) and is case-insensitive; anything
    that is not exactly six hex digits (plus the optional prefix) returns
    ``None`` so callers can ignore mid-typing and junk input without
    disturbing the current selection.
    """
    if type(value) is not str:
        return None
    text = value.strip()
    if not text:
        return None
    text = text.removeprefix("#")
    if len(text) != 6:
        return None
    normalized = f"#{text.lower()}"
    return normalized if is_valid_console_appearance_color(normalized) else None


@dataclass(frozen=True, slots=True)
class ConsoleConversationAppearance:
    """Cosmetic icon + color owned by one Console conversation."""

    icon: str | None = None
    color: str | None = None

    def __post_init__(self) -> None:
        if self.icon is not None and not is_valid_console_appearance_icon(
            self.icon
        ):
            raise ValueError("icon must be a single grapheme of width <= 2.")
        if self.color is not None and not is_valid_console_appearance_color(
            self.color
        ):
            raise ValueError("color must be a canonical #rrggbb hex string.")


def parse_console_conversation_appearance(
    metadata: object,
) -> ConsoleConversationAppearance:
    """Parse appearance metadata, sanitizing malformed members to unset."""
    outer = _metadata_object(metadata)
    owned = outer.get(CONSOLE_APPEARANCE_METADATA_KEY)
    if not isinstance(owned, Mapping) or set(owned) != {"icon", "color"}:
        return ConsoleConversationAppearance()
    icon = owned["icon"]
    color = owned["color"]
    return ConsoleConversationAppearance(
        icon=icon if is_valid_console_appearance_icon(icon) else None,
        color=color if is_valid_console_appearance_color(color) else None,
    )


def merge_console_conversation_appearance(
    metadata: object,
    appearance: ConsoleConversationAppearance,
) -> dict[str, object]:
    """Replace only the Console appearance key, preserving siblings."""
    if not isinstance(appearance, ConsoleConversationAppearance):
        raise TypeError("appearance must be ConsoleConversationAppearance.")
    # Reconstruct to reject objects forged by bypassing the frozen dataclass API.
    validated = ConsoleConversationAppearance(
        icon=appearance.icon, color=appearance.color
    )
    merged = _metadata_object(metadata)
    merged[CONSOLE_APPEARANCE_METADATA_KEY] = {
        "icon": validated.icon,
        "color": validated.color,
    }
    return merged


def _metadata_object(metadata: object) -> dict[str, Any]:
    if isinstance(metadata, Mapping):
        return dict(metadata)
    if type(metadata) is not str or not metadata:
        return {}
    try:
        decoded = json.loads(metadata)
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}
