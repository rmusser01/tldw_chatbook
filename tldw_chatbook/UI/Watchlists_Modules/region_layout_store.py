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

#: What to show before anyone has ever touched collapse state. Through Phase
#: C, CONTENT held only a placeholder stub, so it started collapsed rather
#: than spending a third of the centre column on "Reader arrives in the next
#: slice." on every first launch. Phase D wires a real reader into CONTENT
#: (`ContentPane`), so that reasoning no longer applies -- a first-run user
#: should see the reader like every other region. `RegionLayout()` (nothing
#: collapsed) is now both the first-run default AND what a genuinely empty
#: persisted layout means; see the None-vs-`[]` handling in
#: `load_region_layout` below for why the *distinction between those two
#: cases* still matters even though they resolve to the same value today.
_FIRST_RUN_DEFAULT = RegionLayout()

#: One-time-migration marker (Phase D). A user who saved ANY layout before
#: this change necessarily has CONTENT in their persisted `collapsed_regions`
#: unless they specifically expanded it: CONTENT started collapsed by
#: default (see `_FIRST_RUN_DEFAULT`'s history above), so every save made
#: while that default was in effect carried CONTENT along, whether or not
#: the user meant anything by it -- there was nothing behind it to look at.
#: Honoring that persisted membership forever would leave every such user's
#: reader stuck collapsed post-upgrade, looking broken rather than shipped.
#: This key is set exactly once, the first time `load_region_layout` runs
#: after upgrading: while unset, a persisted CONTENT collapse is dropped
#: (never a persisted expansion -- only ever a discard); once set, a later
#: DELIBERATE collapse of CONTENT (the reader now exists and can genuinely
#: be closed on purpose) is honored like any other region on every load
#: after that, including a value re-added after the marker was set.
_CONTENT_READER_MIGRATED_KEY = "content_reader_migrated"


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

    Also performs the Phase D one-time migration (see
    `_CONTENT_READER_MIGRATED_KEY`): the first time this runs after
    upgrading, any persisted CONTENT collapse is dropped, since it can only
    be a leftover from the placeholder-stub-era default rather than a
    decision about the real reader. Every call after that honors CONTENT's
    collapse state exactly like any other region's.

    Returns:
        The collapse state to apply: the first-run default
        (`_FIRST_RUN_DEFAULT`) when the config key has never been saved, or
        the persisted collapsed set otherwise (silently dropping any
        unrecognized region names or non-sequence stored value), with the
        one-time CONTENT migration applied when it has not run yet.
    """
    migrated = bool(get_cli_setting(_SECTION, _CONTENT_READER_MIGRATED_KEY, False))
    raw = get_cli_setting(_SECTION, _KEY, None)

    if raw is None:
        if not migrated:
            _mark_content_reader_migrated()
        return _FIRST_RUN_DEFAULT
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence):
        logger.debug("Ignoring non-sequence watchlists collapse state: {!r}", raw)
        if not migrated:
            _mark_content_reader_migrated()
        return RegionLayout()

    collapsed = set()
    for value in raw:
        try:
            collapsed.add(Region(str(value)))
        except ValueError:
            logger.debug("Ignoring unknown watchlists region {!r} from config.", value)

    if not migrated:
        # Stub-era saves always carried CONTENT along in `collapsed` (it was
        # the default), so its presence here says nothing about user intent.
        # Drop it exactly once; `_mark_content_reader_migrated` ensures a
        # DELIBERATE re-collapse afterward is never touched again.
        collapsed.discard(Region.CONTENT)
        _mark_content_reader_migrated()

    return RegionLayout(collapsed=frozenset(collapsed))


def _mark_content_reader_migrated() -> None:
    """Record that the Phase D CONTENT-collapse migration has run.

    Best-effort, matching `save_region_layout`'s own error handling: failing
    to persist the marker means the next launch re-runs (and re-no-ops, since
    it only ever discards) the migration rather than losing collapse state.
    """
    try:
        save_setting_to_cli_config(_SECTION, _CONTENT_READER_MIGRATED_KEY, True)
    except Exception:
        logger.opt(exception=True).debug(
            "Failed to persist watchlists content-reader migration marker."
        )


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
