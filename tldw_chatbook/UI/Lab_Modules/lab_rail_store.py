"""Persist the Lab frame's rail collapse state to the user's config.

Collapse is a UI preference, not data, so it belongs in config. It cannot
live in ``BaseAppScreen.save_state``: navigation builds a fresh screen
instance every time (``app.py`` ``_create_navigation_screen``), so
screen-scoped state does not survive a mode switch.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ...config import get_cli_setting, save_setting_to_cli_config
from .lab_rail_layout import LAB_RAIL_INSPECTOR, LAB_RAILS, LabRailLayout

logger = logger.bind(module="LabRailStore")

#: Flat config section. Both flat and dotted sections round-trip correctly,
#: so flat is chosen simply for consistency with the sibling Watchlists store.
LAB_CONFIG_SECTION = "lab"
LAB_COLLAPSED_RAILS_KEY = "collapsed_rails"

#: What to show before anyone has touched collapse state. The left rail is the
#: mode's primary navigation and earns its width; the inspector starts closed.
LAB_FIRST_RUN_LAYOUT = LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR}))


def load_rail_layout() -> LabRailLayout:
    """Read collapse state from config.

    Distinguishes "never saved" from "saved as empty": ``get_cli_setting``
    returns its ``default`` only when the key is absent, so passing ``None``
    -- not ``[]`` -- lets a genuinely unset key be told apart from a user who
    explicitly expanded everything. Collapsing that distinction would
    re-impose the first-run default on every session.

    Returns:
        The stored layout, or :data:`LAB_FIRST_RUN_LAYOUT` when unset or
        unreadable.
    """
    raw: Any = get_cli_setting(LAB_CONFIG_SECTION, LAB_COLLAPSED_RAILS_KEY, None)
    if raw is None:
        return LAB_FIRST_RUN_LAYOUT
    if not isinstance(raw, list):
        logger.warning(
            "Ignoring non-list {}.{} value {!r}; using the first-run layout.",
            LAB_CONFIG_SECTION,
            LAB_COLLAPSED_RAILS_KEY,
            raw,
        )
        return LAB_FIRST_RUN_LAYOUT
    known = {value for value in raw if isinstance(value, str) and value in LAB_RAILS}
    return LabRailLayout(collapsed=frozenset(known))


def save_rail_layout(layout: LabRailLayout) -> None:
    """Write collapse state to config.

    Args:
        layout: Layout to persist. Stored as a sorted list of rail names,
            since TOML cannot hold a frozenset and sorting keeps the config
            file's diff stable.
    """
    save_setting_to_cli_config(
        LAB_CONFIG_SECTION,
        LAB_COLLAPSED_RAILS_KEY,
        sorted(layout.collapsed),
    )
