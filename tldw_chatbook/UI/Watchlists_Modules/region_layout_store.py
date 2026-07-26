"""Persist Watchlists workbench collapse state to the user's config.

Collapse state is UI preference, not data, so it belongs in config rather
than SubscriptionsDB. Solo is deliberately not persisted — it is a transient
view mode, and restoring it across restarts would strand the user in a
layout they did not choose. While a layout is soloed, `save_region_layout`
writes the *pre-solo baseline* (`RegionLayout.collapsed_for_persistence`),
not the solo-derived collapsed set: the latter is what the other centre
panes look like while isolating the soloed one, not something the user
configured, and persisting it would leave a restart with no baseline left
to recover the user's actual layout from.
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

#: What to show before anyone has ever touched collapse state. CONTENT's
#: reader is a Phase D stub, so it starts collapsed rather than spending a
#: third of the centre column on "Reader arrives in the next slice." on
#: every first launch. This is deliberately NOT the same value as
#: `RegionLayout()` (nothing collapsed) — see the None-vs-`[]` handling in
#: `load_region_layout` below for why the distinction matters.
_FIRST_RUN_DEFAULT = RegionLayout(collapsed=frozenset({Region.CONTENT}))


def load_region_layout() -> RegionLayout:
    """Read collapse state from config.

    Distinguishes "the key has never been saved" from "the key was saved as
    an empty list": `get_cli_setting` returns its `default` argument only
    when the key is absent from config, so passing `None` — not `[]` — lets
    a genuinely-unset key be told apart from a user who explicitly
    re-expanded everything and had that saved. Collapsing that distinction
    (as an earlier version of this function did, defaulting to `[]`) means
    every session of every user who has never specifically touched CONTENT
    gets the placeholder stub forever, not just on a true first run — and a
    heuristic that treats "empty" as "apply the first-run default" would
    permanently strand a user who deliberately keeps CONTENT expanded, since
    saving that exact choice looks identical to never having saved anything.

    Returns:
        The collapse state to apply: the first-run default
        (`_FIRST_RUN_DEFAULT`) when the config key has never been saved, or
        the persisted collapsed set otherwise (silently dropping any
        unrecognized region names or non-sequence stored value).
    """
    raw = get_cli_setting(_SECTION, _KEY, None)
    if raw is None:
        return _FIRST_RUN_DEFAULT
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
    """Write collapse state to config. Solo state is not persisted.

    Args:
        layout: The layout to persist. Only its (solo-resolved) collapsed
            set is written — see `RegionLayout.collapsed_for_persistence`;
            `solo_region` and the pre-solo baseline itself never round-trip
            through config.
    """
    values = sorted(region.value for region in layout.collapsed_for_persistence())
    try:
        save_setting_to_cli_config(_SECTION, _KEY, values)
    except Exception:
        logger.opt(exception=True).debug("Failed to persist watchlists collapse state.")
