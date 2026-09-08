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
    LIBRARY_EMERGENCY_WIDTH,
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

    One frozen profile per reader destination. Every field is a width in
    terminal cells, and the two task-31633 fields are opt-in: their defaults
    reproduce the pre-task behaviour exactly, so a destination that does not
    name them is unaffected.

    Attributes:
        list_min_width: Floor for the list (Items) pane. The resolver
            collapses the pane rather than paint it narrower.
        list_target_width: Not read by the resolver. The list's automatic
            width comes from ``preferences.items_width`` (defaulted from
            ``ITEMS_TARGET_WIDTH``); the field is retained only so profiles
            can be constructed with it in tests.
        list_comfort_width: Ceiling for ``list_grows``. Surplus width stops
            flowing into the list here and goes to the work pane instead.
        list_max_width: Hard ceiling for the list pane, including a width the
            user typed.
        work_min_width: Floor for the work (Reader) pane; below it the shell
            drops the list pane rather than squeeze the document.
        work_comfort_width: Not read by the resolver. The ``list_grows``
            gate is ``max(work_min_width, READER_COMFORT_WIDTH)``; only
            Collections sets this field (56), to no effect.
        list_grows: When ``True``, a work pane already at
            ``max(work_min_width, READER_COMFORT_WIDTH)`` shares half of any
            further surplus with the list, up to ``list_comfort_width``,
            instead of absorbing every extra cell.
            Automatic widths only: a custom width is obeyed as typed. Default
            ``False`` (every extra cell goes to the work pane); only Media
            opts in today.
        list_first_when_empty: When ``True``, a width too narrow to seat the
            list beside the work pane keeps the LIST and gives the work pane
            what is left, rather than dropping the list for a work pane that
            has nothing in it. Only applies while ``reader_has_item`` is
            ``False``; opening an item hands the stage straight back. Default
            ``False`` (the pre-task-32065 behaviour); only Media opts in
            today, where 60x24 painted "Select a media item to read it here."
            with no list to select from.
        grip_width: Width of EACH of the two pane grips -- both what a grip
            paints and what the resolver holds back for it, so the two can
            never disagree (the resolved layout carries this width to the
            shell). Defaults to ``PANE_GRIP_WIDTH`` (5). Media passes 1: its
            two five-column grips left ten dead columns around the Items pane
            (task-31633 AC#2), and task-31951 opted Conversations, Skills and
            Collections in for the same reason. A grip narrower than four
            cells paints the one-cell guillemet instead of the ``<---`` run.
    """

    list_min_width: int = 32
    list_target_width: int = 40
    list_comfort_width: int = 56
    list_max_width: int = 72
    work_min_width: int = 44
    work_comfort_width: int = 44
    list_grows: bool = False
    list_first_when_empty: bool = False
    grip_width: int = PANE_GRIP_WIDTH


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
    """One rendered layout derived from preferences and available width.

    Attributes:
        library_open: Whether the library (rail) pane is rendered. Starts
            from the user's preference and is forced ``False`` when the
            width cannot seat it and still leave ``work_min_width``.
        items_open: Whether the items (list) pane is rendered, under the
            same preference-then-width rule as ``library_open``.
        library_width: Columns given to the library pane, ``0`` when it is
            closed.
        items_width: Columns given to the items pane, ``0`` when it is
            closed.
        reader_width: Columns left for the work (Reader) pane -- the
            remainder, never negative.
        priority_pane: Which pane a width-starved layout kept open
            (``"library"``/``"items"``), or ``None`` when nothing had to
            be dropped.
        grip_width: The resolving profile's per-grip width in columns,
            stamped here so the shell paints both grips from the same
            number the resolver reserved (task-31952).

    The three widths are what each pane actually gets to paint: the pane
    GRIPS are already deducted. The resolver holds back ``2 * grip_width``
    before dividing what is left, so ``library_width + items_width +
    reader_width`` plus that reserve is the full terminal width.

    ``grip_width`` carries the resolving profile's per-grip width through to
    the shell, which sizes both grips from it (task-31952 AC#3): the columns
    the resolver held back and the columns a grip paints are then literally
    the same number, and a destination cannot reserve five and paint one.
    """

    library_open: bool
    items_open: bool
    library_width: int
    items_width: int
    reader_width: int
    priority_pane: PaneName | None
    grip_width: int = PANE_GRIP_WIDTH


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
    reader_has_item: bool = True,
) -> AdaptiveReaderEffectiveLayout:
    """Resolve saved pane preferences into one responsive effective layout.

    Args:
        width: Available shell width in terminal cells.
        preferences: Persisted manual pane preferences.
        profile: Destination list and work-pane width policy.
        previous: Previously resolved layout used for hysteresis.
        priority: Pane explicitly requested by the user, if any.
        reader_has_item: Whether the work (Reader) pane has an item open.
            When ``False`` (the pane shows only its empty-state placeholder)
            the width normally reserved for a document is given to the Items
            list instead, down to the work pane's floor, so a long title stops
            truncating in the wasted space (task-31979). Defaults to ``True``,
            which reproduces the pre-task split exactly; only the non-starved
            main path reallocates -- a width-starved priority layout has no
            surplus to give. Automatic widths only: obeyed as typed under
            ``custom_widths_enabled``, matching the ``list_grows`` gate.
            It also decides the stage below the single-stage floor for a
            profile with ``list_first_when_empty``: with nothing to read, the
            list is kept instead of the empty work pane (task-32065).

    Returns:
        Current effective pane geometry.

    Raises:
        TypeError: If ``preferences`` or ``profile`` has the wrong type, or
            ``reader_has_item`` is not a boolean.
        ValueError: If ``width`` is not a non-negative integer or ``priority``
            is unsupported.
    """
    if type(width) is not int or width < 0:
        raise ValueError("width must be a non-negative integer.")
    if not isinstance(preferences, AdaptiveReaderLayoutPreferences):
        raise TypeError("preferences must be AdaptiveReaderLayoutPreferences.")
    if not isinstance(profile, AdaptiveReaderLayoutProfile):
        raise TypeError("profile must be AdaptiveReaderLayoutProfile.")
    if type(reader_has_item) is not bool:
        raise TypeError("reader_has_item must be a boolean.")
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
            grip_width=profile.grip_width,
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

    grip_width = 2 * profile.grip_width
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
                grip_width=profile.grip_width,
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

    if (
        not items_open
        and not reader_has_item
        and preferences.items_open
        and profile.list_first_when_empty
        and width < LIBRARY_EMERGENCY_WIDTH
        and width - grip_width >= profile.list_min_width
    ):
        # task-32065: below the width that seats a list beside the work pane,
        # dropping the list leaves a pane with NOTHING in it as the whole
        # stage -- live at 60x24 Media painted "Select a media item to read
        # it here." over two collapsed-pane grips, with no list to select
        # from and no way back to the rail. With nothing to read, the list
        # wins the stage; opening an item (``reader_has_item=True``) resolves
        # the ordinary way and hands it straight back.
        #
        # Bounded to the ordinary single-stage floor (64 -- the same constant
        # the rail-and-canvas layouts use) on purpose: at 80x24 the Items
        # pane is deliberately dropped and focus evacuates to its grip, and
        # three tests pin that. Below the floor nothing else is on screen at
        # all, which is the case this branch exists for.
        items_width = min(
            max(profile.list_min_width, preferences.items_width),
            width - grip_width,
        )
        return AdaptiveReaderEffectiveLayout(
            library_open=False,
            items_open=True,
            library_width=0,
            items_width=items_width,
            reader_width=max(width - grip_width - items_width, 0),
            priority_pane=None,
            grip_width=profile.grip_width,
        )

    library_width = requested_library_width if library_open else 0
    items_width = preferences.items_width if items_open else 0
    if items_open and not library_open:
        # task-31953: this clamp is deliberately NOT gated on
        # `custom_widths_enabled` -- with the Library pane gone a typed 32
        # still widens to the comfort ceiling (52 at width 100 on Media), as
        # does the priority-pane clamp above. Documented, not changed; see
        # `test_a_typed_custom_items_width_is_still_widened_once_the_library_closes`.
        comfort_width = max(
            items_width,
            min(profile.list_comfort_width, profile.list_max_width),
        )
        items_width = min(
            comfort_width,
            max(width - grip_width - work_min_width, items_width),
        )
    if items_open and profile.list_grows and not preferences.custom_widths_enabled:
        # task-31633: past this point every remaining cell used to go to the
        # Reader, so a 235-cell terminal painted a NARROWER list than a
        # 100-cell one. Split the Reader's surplus once it is comfortable, up
        # to the same comfort ceiling the library-closed branch above uses.
        #
        # The floor is the Reader's OWN minimum, not READER_COMFORT_WIDTH: a
        # literal 44 would leave Media's Reader on 45 cells at width 100,
        # below the work_min_width=46 that every open/close and hysteresis
        # decision in required_width() was computed against. It is never
        # below READER_COMFORT_WIDTH, so the intent holds for profiles at or
        # under 44.
        #
        # Custom widths are a hand-typed number in Settings: "Automatic"
        # adapts, "Custom" obeys, so growth is off in that mode entirely.
        reader_floor = max(work_min_width, READER_COMFORT_WIDTH)
        surplus = width - grip_width - library_width - items_width - reader_floor
        if surplus > 0:
            items_width = min(
                items_width + surplus // 2,
                max(
                    min(profile.list_comfort_width, profile.list_max_width),
                    items_width,
                ),
            )
    if items_open and not reader_has_item and not preferences.custom_widths_enabled:
        # task-31979: the work (Reader) pane is showing only its empty-state
        # placeholder, so the width normally held for a document is wasted
        # while the Items list truncates long titles into ~56 cells. Hand that
        # freed width to the list, down to the work pane's own floor; selecting
        # an item (reader_has_item=True) restores the split unchanged. Gated on
        # automatic widths like the list_grows block above: Custom obeys.
        freed = width - grip_width - library_width - items_width - work_min_width
        if freed > 0:
            items_width += freed
    return AdaptiveReaderEffectiveLayout(
        library_open=library_open,
        items_open=items_open,
        library_width=library_width,
        items_width=items_width,
        reader_width=max(width - grip_width - library_width - items_width, 0),
        priority_pane=priority,
        grip_width=profile.grip_width,
    )
