"""Pure responsive-layout state shared by Library adaptive readers.

Layering (TASK-22223): this module is a config-safe leaf. `config.py`'s
`_load_settings_uncached` normalizes persisted reader preferences through
`normalize_adaptive_reader_preferences` at config-module import, so this
module must stay importable without executing any feature package: stdlib
imports only, and it must live under a package whose `__init__` has no side
effects (`Utils/__init__.py` is empty). It previously lived at
`Library/library_adaptive_reader_state.py`, where the `Library` package
`__init__` dragged the collections/tool service stack -- and a live import
cycle through `runtime_policy.bootstrap` -- into every config import.
Guarded by `Tests/Packaging/test_config_import_closure.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .library_rail_width import (
    LIBRARY_CUSTOM_MAX_WIDTH,
    LIBRARY_MIN_WIDTH,
    LIBRARY_REFERENCE_WIDTH,
    project_default_library_width,
)

LIBRARY_TARGET_WIDTH = LIBRARY_REFERENCE_WIDTH
LIBRARY_MAX_WIDTH = LIBRARY_CUSTOM_MAX_WIDTH
ITEMS_TARGET_WIDTH = 40
ITEMS_MIN_WIDTH = 32
ITEMS_MAX_WIDTH = 72
READER_COMFORT_WIDTH = 44
PANE_GRIP_WIDTH = 5
LAYOUT_HYSTERESIS_WIDTH = 4

PaneName = Literal["library", "items"]


@dataclass(frozen=True)
class AdaptiveReaderLayoutProfile:
    """Destination-specific list and work-pane width policy.

    ``list_grows`` is opt-in per destination (task-31633): when set, a
    comfortable Reader shares its surplus width with the list instead of
    absorbing every extra cell. Only Media opts in today.
    """

    list_min_width: int = 32
    list_target_width: int = 40
    list_comfort_width: int = 56
    list_max_width: int = 72
    work_min_width: int = 44
    work_comfort_width: int = 44
    list_grows: bool = False


@dataclass(frozen=True)
class AdaptiveReaderLayoutPreferences:
    """Persisted manual pane choices and normalized target widths."""

    library_open: bool = True
    items_open: bool = True
    custom_widths_enabled: bool = False
    library_width: int = LIBRARY_TARGET_WIDTH
    items_width: int = ITEMS_TARGET_WIDTH


@dataclass(frozen=True)
class AdaptiveReaderEffectiveLayout:
    """One rendered layout derived from preferences and available width."""

    library_open: bool
    items_open: bool
    library_width: int
    items_width: int
    reader_width: int
    priority_pane: PaneName | None


def _coerce_bool(value: Any, default: bool) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _coerce_width(value: Any, default: int, minimum: int, maximum: int) -> int:
    if type(value) is int:
        width = value
    elif isinstance(value, str):
        try:
            width = int(value.strip())
        except ValueError:
            return default
    else:
        return default
    return min(max(width, minimum), maximum)


def normalize_adaptive_reader_preferences(
    raw: Mapping[str, Any],
) -> AdaptiveReaderLayoutPreferences:
    """Normalize persisted values without importing application configuration.

    Args:
        raw: Untrusted persisted preference values.

    Returns:
        Normalized pane-open and width preferences.
    """
    library_open = _coerce_bool(raw.get("library_open"), True)
    items_open = _coerce_bool(raw.get("items_open"), True)
    custom_widths_enabled = _coerce_bool(raw.get("custom_widths_enabled"), False)
    if not custom_widths_enabled:
        return AdaptiveReaderLayoutPreferences(
            library_open=library_open,
            items_open=items_open,
        )
    return AdaptiveReaderLayoutPreferences(
        library_open=library_open,
        items_open=items_open,
        custom_widths_enabled=True,
        library_width=_coerce_width(
            raw.get("library_width"),
            LIBRARY_TARGET_WIDTH,
            LIBRARY_MIN_WIDTH,
            LIBRARY_MAX_WIDTH,
        ),
        items_width=_coerce_width(
            raw.get("items_width"),
            ITEMS_TARGET_WIDTH,
            ITEMS_MIN_WIDTH,
            ITEMS_MAX_WIDTH,
        ),
    )


def resolve_adaptive_reader_layout(
    width: int,
    preferences: AdaptiveReaderLayoutPreferences,
    profile: AdaptiveReaderLayoutProfile,
    *,
    previous: AdaptiveReaderEffectiveLayout | None = None,
    priority: PaneName | None = None,
) -> AdaptiveReaderEffectiveLayout:
    """Resolve saved pane preferences into one responsive effective layout.

    Args:
        width: Available shell width in terminal cells.
        preferences: Persisted manual pane preferences.
        profile: Destination list and work-pane width policy.
        previous: Previously resolved layout used for hysteresis.
        priority: Pane explicitly requested by the user, if any.

    Returns:
        Current effective pane geometry.

    Raises:
        TypeError: If ``preferences`` or ``profile`` has the wrong type.
        ValueError: If ``width`` is not a non-negative integer or ``priority``
            is unsupported.
    """
    if type(width) is not int or width < 0:
        raise ValueError("width must be a non-negative integer.")
    if not isinstance(preferences, AdaptiveReaderLayoutPreferences):
        raise TypeError("preferences must be AdaptiveReaderLayoutPreferences.")
    if not isinstance(profile, AdaptiveReaderLayoutProfile):
        raise TypeError("profile must be AdaptiveReaderLayoutProfile.")
    if priority not in {None, "library", "items"}:
        raise ValueError("priority must be library, items, or None.")
    if width == 0:
        return AdaptiveReaderEffectiveLayout(
            library_open=False,
            items_open=False,
            library_width=0,
            items_width=0,
            reader_width=0,
            priority_pane=None,
        )

    requested_library_width = (
        preferences.library_width
        if preferences.custom_widths_enabled
        else project_default_library_width(width)
    )
    if priority is None and previous is not None:
        inherited = previous.priority_pane
        if (
            inherited == "library"
            and preferences.library_open
            or inherited == "items"
            and preferences.items_open
        ):
            priority = inherited

    grip_width = 2 * PANE_GRIP_WIDTH
    work_min_width = max(profile.work_min_width, 0)
    library_open = preferences.library_open
    items_open = preferences.items_open
    if priority is not None:
        if priority == "library":
            library_open = True
        else:
            items_open = True

        full_width = (
            grip_width
            + (requested_library_width if library_open else 0)
            + (preferences.items_width if items_open else 0)
            + work_min_width
        )
        if width < full_width:
            if priority == "library":
                items_open = False
                library_width = (
                    requested_library_width
                    if width >= grip_width + requested_library_width + work_min_width
                    else min(LIBRARY_MIN_WIDTH, max(width - grip_width, 0))
                )
                items_width = 0
            else:
                library_open = False
                library_width = 0
                items_width = (
                    preferences.items_width
                    if width >= grip_width + preferences.items_width + work_min_width
                    else min(
                        max(profile.list_min_width, 0),
                        max(width - grip_width, 0),
                    )
                )
                items_width = min(
                    max(
                        items_width,
                        min(profile.list_comfort_width, profile.list_max_width),
                    ),
                    max(width - grip_width - work_min_width, items_width),
                )
            return AdaptiveReaderEffectiveLayout(
                library_open=library_open,
                items_open=items_open,
                library_width=library_width,
                items_width=items_width,
                reader_width=max(width - grip_width - library_width - items_width, 0),
                priority_pane=priority,
            )
        priority = None

    def required_width(open_library: bool, open_items: bool) -> int:
        return (
            grip_width
            + (requested_library_width if open_library else 0)
            + (preferences.items_width if open_items else 0)
            + work_min_width
        )

    if width < required_width(library_open, items_open):
        library_open = False
    if width < required_width(library_open, items_open):
        items_open = False

    if previous is not None:
        nominal_width = required_width(library_open, items_open)
        if (
            library_open
            and not previous.library_open
            and width < nominal_width + LAYOUT_HYSTERESIS_WIDTH
        ):
            library_open = False
        if (
            items_open
            and not previous.items_open
            and width
            < required_width(library_open, items_open) + LAYOUT_HYSTERESIS_WIDTH
        ):
            items_open = False

    library_width = requested_library_width if library_open else 0
    items_width = preferences.items_width if items_open else 0
    if items_open and not library_open:
        comfort_width = max(
            items_width,
            min(profile.list_comfort_width, profile.list_max_width),
        )
        items_width = min(
            comfort_width,
            max(width - grip_width - work_min_width, items_width),
        )
    if items_open and profile.list_grows:
        # task-31633: past this point every remaining cell used to go to the
        # Reader, so a 235-cell terminal painted a NARROWER list than a
        # 100-cell one. Split the Reader's surplus once it is comfortable --
        # its floor is its own minimum, which for a destination whose reader
        # needs more than READER_COMFORT_WIDTH is what "comfortable" means,
        # and which leaves the compact allocations byte-identical.
        reader_floor = max(work_min_width, READER_COMFORT_WIDTH)
        surplus = width - grip_width - library_width - items_width - reader_floor
        if surplus > 0:
            items_width = min(
                items_width + surplus // 2,
                max(profile.list_max_width, items_width),
            )
    return AdaptiveReaderEffectiveLayout(
        library_open=library_open,
        items_open=items_open,
        library_width=library_width,
        items_width=items_width,
        reader_width=max(width - grip_width - library_width - items_width, 0),
        priority_pane=priority,
    )
