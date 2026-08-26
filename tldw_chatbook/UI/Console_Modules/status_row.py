"""Status-row (chips strip) placement and collapse preferences (task-17652).

The Console status chips either top the composer cluster (the default —
owner ruling 2026-08-17) or close the shell underneath it, the pre-17652
order from TASK-15704. The preference lives at ``[console]
status_chips_position``; the strip's collapse state persists at
``[console] status_chips_collapsed``. ChatScreen composes from these
resolvers and re-applies the position on screen resume, because the
screen instance is cached across navigation and a Settings change must
take effect without a recompose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from loguru import logger

if TYPE_CHECKING:
    from textual.widget import Widget

STATUS_CHIPS_POSITION_ABOVE = "above"
STATUS_CHIPS_POSITION_BELOW = "below"
DEFAULT_STATUS_CHIPS_POSITION = STATUS_CHIPS_POSITION_ABOVE


def _console_section(app_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the ``[console]`` mapping from an app-config tree, or ``{}``."""
    if not isinstance(app_config, Mapping):
        return {}
    console_config = app_config.get("console")
    return console_config if isinstance(console_config, Mapping) else {}


def resolve_status_chips_position(app_config: Mapping[str, Any] | None) -> str:
    """Return ``"above"`` or ``"below"`` for the status-row placement.

    Args:
        app_config: The application config mapping (``[console]
            status_chips_position`` is the governing key).

    Returns:
        The validated position; anything unrecognized falls back to the
        default rather than propagating a typo into compose order.
    """
    raw = _console_section(app_config).get("status_chips_position")
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in (STATUS_CHIPS_POSITION_ABOVE, STATUS_CHIPS_POSITION_BELOW):
            return value
    return DEFAULT_STATUS_CHIPS_POSITION


def resolve_status_chips_collapsed(app_config: Mapping[str, Any] | None) -> bool:
    """Return the persisted Status-row collapse preference (default expanded)."""
    return bool(_console_section(app_config).get("status_chips_collapsed", False))


def poke_console_setting(app_config: Any, key: str, value: Any) -> None:
    """Update a ``[console]`` value in the live in-memory config tree.

    ADR-020-style immediate effect: the app captures ``app_config`` once at
    startup, so a disk write alone would not take effect until restart.
    Both the working ``console`` section and the raw comprehensive tree are
    updated so every reader agrees.
    """
    if not isinstance(app_config, dict):
        return
    console_section = app_config.get("console")
    if not isinstance(console_section, dict):
        console_section = {}
        app_config["console"] = console_section
    console_section[key] = value
    raw = app_config.get("COMPREHENSIVE_CONFIG_RAW")
    if isinstance(raw, dict):
        raw.setdefault("console", {})[key] = value


def apply_status_chips_position(screen: Any) -> bool:
    """Move the mounted chips strip to the configured side of the composer.

    "Above" means directly above the composer's top gap — BELOW the
    transient strips (staged evidence, prompt queue), which sit at the top
    of the control deck at all times (task-17661). "Below" restores the
    TASK-15704 bottom row. Never raises: a screen mid-teardown or a strip
    not yet mounted is a no-op, not an error.

    Args:
        screen: The mounted ChatScreen (anything exposing ``query_one`` and
            ``app_instance``).

    Returns:
        True when the strip actually moved.
    """
    try:
        chips = screen.query_one("#console-status-chips")
        composer = screen.query_one("#console-native-composer")
    except Exception:
        return False
    parent = chips.parent
    if parent is None or parent is not composer.parent:
        return False
    position = resolve_status_chips_position(
        getattr(getattr(screen, "app_instance", None), "app_config", None)
    )
    children: list["Widget"] = list(parent.children)
    try:
        chips_index = children.index(chips)
        composer_index = children.index(composer)
        if position == STATUS_CHIPS_POSITION_ABOVE:
            if chips_index == composer_index - 1:
                return False
            parent.move_child(chips, before=composer)
        else:
            if chips_index == composer_index + 1:
                return False
            parent.move_child(chips, after=composer)
    except Exception:
        logger.opt(exception=True).warning("status_chips_position_apply_failed")
        return False
    return True


def persist_status_chips_collapsed(collapsed: bool) -> None:
    """Best-effort disk write for the collapse preference.

    Safe to run on a worker thread; a config-write hiccup logs instead of
    raising, because an uncaught exception in a Textual worker is fatal to
    the app by default.
    """
    from ...config import save_setting_to_cli_config

    try:
        save_setting_to_cli_config("console", "status_chips_collapsed", collapsed)
    except Exception:
        logger.warning("Failed to persist status_chips_collapsed.")
