"""Persist Watchlists workbench collapse state to the user's config.

Collapse state is UI preference, not data, so it belongs in config rather
than SubscriptionsDB. Solo is deliberately not persisted — it is a transient
view mode, and restoring it across restarts would strand the user in a
layout they did not choose.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from ...config import get_cli_setting, save_setting_to_cli_config
from .region_layout import Region, RegionLayout


logger = logger.bind(module="WatchlistsRegionLayoutStore")

#: Flat section. `get_cli_setting` does NOT resolve dotted sections — passing
#: "watchlists.layout" silently returns the default and the setting never
#: round-trips. This repo has shipped that bug before with "chat.images".
_SECTION = "watchlists"
_KEY = "collapsed_regions"


def load_region_layout() -> RegionLayout:
    """Read collapse state from config, defaulting to everything expanded."""
    raw = get_cli_setting(_SECTION, _KEY, [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence):
        logger.debug("Ignoring non-sequence watchlists collapse state: {!r}", raw)
        return RegionLayout()

    collapsed = set()
    for value in raw:
        try:
            collapsed.add(Region(str(value)))
        except ValueError:
            logger.debug("Ignoring unknown watchlists region {!r} from config.", value)
    return RegionLayout(collapsed=frozenset(collapsed))


def save_region_layout(layout: RegionLayout) -> None:
    """Write collapse state to config. Solo state is not persisted."""
    values = sorted(region.value for region in layout.collapsed)
    try:
        save_setting_to_cli_config(_SECTION, _KEY, values)
    except Exception:
        logger.opt(exception=True).debug("Failed to persist watchlists collapse state.")
