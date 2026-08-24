"""Persist the preferred Watchlists side-pane layout in user config."""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from ...config import get_cli_setting, save_settings_to_cli_config
from .region_layout import COLLAPSIBLE_REGIONS, Region, RegionLayout


logger = logger.bind(module="WatchlistsRegionLayoutStore")

LAYOUT_VERSION = 2

_SECTION = "watchlists"
_COLLAPSED_REGIONS_KEY = "collapsed_regions"
_LAYOUT_VERSION_KEY = "layout_version"
_RETIRED_MIGRATION_KEY = "content_reader_migrated"

_FIRST_RUN_DEFAULT = RegionLayout(collapsed=frozenset({Region.RIGHT_RAIL}))


def _normalize_values(raw: object) -> list[str]:
    """Return canonical persisted values for a raw config value."""
    if raw is None:
        return [
            region.value
            for region in COLLAPSIBLE_REGIONS
            if region in _FIRST_RUN_DEFAULT.collapsed
        ]
    if isinstance(raw, str):
        raw = [raw]
    elif not isinstance(raw, Sequence):
        logger.debug("Ignoring non-sequence watchlists collapse state: {!r}", raw)
        raw = []

    return [region.value for region in COLLAPSIBLE_REGIONS if region.value in raw]


def _write_values(values: list[str]) -> bool:
    """Atomically persist normalized layout values and retire the old marker."""
    try:
        return save_settings_to_cli_config(
            {
                _SECTION: {
                    _COLLAPSED_REGIONS_KEY: values,
                    _LAYOUT_VERSION_KEY: LAYOUT_VERSION,
                }
            },
            delete_keys={_SECTION: (_RETIRED_MIGRATION_KEY,)},
        )
    except Exception:
        logger.opt(exception=True).debug(
            "Failed to persist normalized watchlists pane layout."
        )
        return False


def load_region_layout() -> RegionLayout:
    """Load and best-effort normalize the preferred side-pane layout.

    Returns:
        A safe layout containing only currently collapsible side panes.
    """
    raw = get_cli_setting(_SECTION, _COLLAPSED_REGIONS_KEY, None)
    version = get_cli_setting(_SECTION, _LAYOUT_VERSION_KEY, None)
    values = _normalize_values(raw)

    if version != LAYOUT_VERSION or raw != values:
        _write_values(values)

    collapsed = frozenset(
        region for region in COLLAPSIBLE_REGIONS if region.value in values
    )
    return RegionLayout(collapsed=collapsed)


def save_region_layout(layout: RegionLayout) -> bool:
    """Persist only the layout's preferred collapsible side panes.

    Args:
        layout: Preferred layout to persist.

    Returns:
        The configuration writer's Boolean result, or ``False`` if it raises.
    """
    values = [
        region.value
        for region in COLLAPSIBLE_REGIONS
        if region in layout.collapsed
    ]
    return _write_values(values)
